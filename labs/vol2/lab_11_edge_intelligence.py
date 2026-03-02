import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full")

# ─────────────────────────────────────────────────────────────────────────────
# LAB V2-11: THE FEDERATION PARADOX
#
# Volume II, Chapter 11 — Edge Intelligence
#
# Core Invariant: Federated learning communication cost, centralized vs federated
#   Federated learning keeps data on device but communicates model updates instead.
#   Communication cost per round = model_size × 2 (upload + download).
#   Without gradient compression, federated communication can exceed centralized
#   data transfer by orders of magnitude — privacy is NOT free.
#
# 2 Contexts:
#   Centralized  — Cloud training on H100 (data travels to server)
#   Federated    — On-device training (gradients travel to server)
#
# Act I  (12–15 min): Federated Communication Cost Revelation
#   Stakeholder: Privacy Architect at a mobile keyboard team
#   Instruments: participating device fraction, model size, compression, rounds/day
#   Prediction: centralized vs federated bandwidth comparison
#   Overlay: predicted ratio vs actual physics
#   Reflection: primary bandwidth reduction technique in production (Gboard)
#
# Act II (20–25 min): Privacy-Utility Tradeoff Designer
#   Stakeholder: Product Lead choosing between 3 deployment options
#   Instruments: DP epsilon, local epochs, participating fraction, aggregation rounds
#   Prediction: which option provides formal privacy guarantees
#   Failure states:
#     - Accuracy drops below utility threshold (kind="danger")
#     - Daily bandwidth exceeds 1 PB (kind="warn")
#   Reflection: what ε=1 means in differential privacy
#
# Design Ledger: saves chapter="v2_11"
# ─────────────────────────────────────────────────────────────────────────────


# ─── CELL 0: SETUP (hide_code=False — leave visible) ─────────────────────────
@app.cell
def _():
    import marimo as mo
    import sys
    import math
    from pathlib import Path
    import plotly.graph_objects as go
    import numpy as np

    _root = Path(__file__).resolve().parents[2]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

    from labs.core.state import DesignLedger
    from labs.core.style import COLORS, LAB_CSS, apply_plotly_theme

    # ── Hardware constants ────────────────────────────────────────────────────
    # All values from @sec-edge-intelligence and NVIDIA/mobile specs

    H100_BW_GBS       = 3350    # GB/s HBM3e — NVIDIA H100 SXM5 spec
    H100_RAM_GB       = 80      # GB HBM3e — NVIDIA spec
    H100_TDP_W        = 700     # Watts TDP — NVIDIA spec

    MOBILE_BW_GBS     = 68      # GB/s mobile NPU memory bandwidth — Apple A17 class
    MOBILE_RAM_GB     = 8       # GB typical smartphone RAM
    MOBILE_NPU_TOPS   = 35      # TOPS INT8 — Apple A16 Neural Engine class

    LTE_UL_MBPS       = 50      # LTE uplink bandwidth per device (Mbps) — avg real-world
    WIFI_UL_MBPS      = 100     # WiFi uplink bandwidth per device (Mbps) — 802.11ac typical

    # ── Keyboard model constants (from edge_intelligence.qmd narrative) ───────
    # @sec-edge-intelligence-federated-learning-6e7e references 1B param model
    # at 4 GB FP32, 2 GB FP16 for keyboard suggestion use case
    KEYBOARD_MODEL_PARAMS_B  = 1.0      # 1B parameter keyboard model
    KEYBOARD_MODEL_FP16_GB   = 2.0      # 2 GB FP16 model size — chapter narrative
    KEYBOARD_MODEL_FP32_GB   = 4.0      # 4 GB FP32 model size — chapter narrative

    # ── Centralized baseline (from chapter text) ──────────────────────────────
    # 100 bytes/keystroke × 1B keystrokes/day = 100 GB/day centralized data
    KEYSTROKES_PER_DAY_B     = 1.0      # 1B keystrokes/day from 100M devices
    BYTES_PER_KEYSTROKE      = 100      # bytes per keystroke (context + metadata)

    ledger = DesignLedger()

    return (
        mo, ledger, COLORS, LAB_CSS, apply_plotly_theme,
        go, np, math,
        H100_BW_GBS, H100_RAM_GB, H100_TDP_W,
        MOBILE_BW_GBS, MOBILE_RAM_GB, MOBILE_NPU_TOPS,
        LTE_UL_MBPS, WIFI_UL_MBPS,
        KEYBOARD_MODEL_PARAMS_B, KEYBOARD_MODEL_FP16_GB, KEYBOARD_MODEL_FP32_GB,
        KEYSTROKES_PER_DAY_B, BYTES_PER_KEYSTROKE,
    )


# ─── CELL 1: HEADER (hide_code=True) ─────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, LAB_CSS, COLORS):
    _federated_color = COLORS["Mobile"]   # orange — federated/edge regime
    _cloud_color     = COLORS["Cloud"]    # indigo — centralized regime
    mo.vstack([
        LAB_CSS,
        mo.Html(f"""
        <div style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 60%, #1a0a20 100%);
                    padding: 36px 44px; border-radius: 16px; color: white;
                    box-shadow: 0 8px 32px rgba(0,0,0,0.35);">
            <div style="font-size: 0.72rem; font-weight: 700; letter-spacing: 0.18em;
                        color: #475569; text-transform: uppercase; margin-bottom: 10px;">
                Machine Learning Systems · Volume II · Lab 11
            </div>
            <h1 style="margin: 0 0 10px 0; font-size: 2.4rem; font-weight: 900;
                       color: #f8fafc; line-height: 1.1; letter-spacing: -0.02em;">
                The Federation Paradox
            </h1>
            <p style="margin: 0 0 22px 0; font-size: 1.05rem; color: #94a3b8;
                      max-width: 640px; line-height: 1.65;">
                Privacy is not free. Federated learning keeps data on-device but moves
                model gradients instead. With 100M devices and a 2 GB model, the
                communication cost per round dwarfs centralized training.
            </p>
            <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                <span class="badge badge-info">Centralized vs Federated</span>
                <span class="badge badge-warn">Communication Cost</span>
                <span class="badge badge-info">Differential Privacy</span>
                <span class="badge badge-ok">35–40 min</span>
            </div>
        </div>
        """),
    ])
    return


# ─── CELL 2: RECOMMENDED READING (hide_code=True) ────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md("""
    **Recommended Reading** — Complete these sections before this lab:

    - **@sec-edge-intelligence-distributed-learning-paradigm-shift-883d** — The Edge Learning Paradigm: centralized vs on-device learning
    - **@sec-edge-intelligence-federated-learning-6e7e** — Federated Learning Algorithms: FedAvg, convergence, communication cost
    - **@sec-edge-intelligence-federated-systems** — Federated Systems at Scale: bandwidth optimization, compression techniques
    - **@sec-edge-intelligence-federated-privacy-a1ed** — Federated Privacy: model inversion attacks, differential privacy, secure aggregation
    """), kind="info")
    return


# ─── CELL 3: CONTEXT TOGGLE (hide_code=True) ─────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    context_toggle = mo.ui.radio(
        options={
            "Centralized (Cloud)": "centralized",
            "Federated (On-Device)": "federated",
        },
        value="Centralized (Cloud)",
        label="Deployment context:",
        inline=True,
    )
    mo.vstack([
        mo.md("### Deployment Context"),
        mo.md("""
Select the training paradigm to compare. This toggle persists across both acts
and colors the metric cards to reflect your chosen regime.
        """),
        context_toggle,
    ])
    return (context_toggle,)


# ─── CELL 4: CONTEXT SPEC CARDS (hide_code=True) ─────────────────────────────
@app.cell(hide_code=True)
def _(mo, context_toggle, COLORS):
    _ctx = context_toggle.value

    _cloud_border  = COLORS["Cloud"]    # indigo
    _fed_border    = COLORS["Mobile"]   # orange
    _active_alpha  = "1.0"
    _passive_alpha = "0.4"

    _cloud_opacity  = _active_alpha if _ctx == "centralized" else _passive_alpha
    _fed_opacity    = _active_alpha if _ctx == "federated"   else _passive_alpha

    mo.Html(f"""
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin: 12px 0;">
        <div style="border: 2px solid {_cloud_border}; border-radius: 12px; padding: 18px;
                    background: #f0f4ff; opacity: {_cloud_opacity};">
            <div style="font-weight: 800; font-size: 0.9rem; color: {_cloud_border};
                        margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.08em;">
                Centralized — Cloud Training
            </div>
            <div style="font-size: 0.83rem; color: #374151; line-height: 1.6;">
                Raw data travels from 100M devices to a central H100 cluster.
                Training happens in one place with full data visibility.<br><br>
                <strong>Privacy cost:</strong> All user keystrokes sent to server<br>
                <strong>Bandwidth cost:</strong> ~100 GB/day of raw data<br>
                <strong>Compute:</strong> Centralized H100s — high utilization
            </div>
        </div>
        <div style="border: 2px solid {_fed_border}; border-radius: 12px; padding: 18px;
                    background: #fff7ed; opacity: {_fed_opacity};">
            <div style="font-weight: 800; font-size: 0.9rem; color: {_fed_border};
                        margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.08em;">
                Federated — On-Device Training
            </div>
            <div style="font-size: 0.83rem; color: #374151; line-height: 1.6;">
                Devices train locally; only model gradients travel to the server.
                Raw data never leaves the device.<br><br>
                <strong>Privacy cost:</strong> Gradient updates may leak via inversion attacks<br>
                <strong>Bandwidth cost:</strong> model_size × 2 × participating_devices<br>
                <strong>Compute:</strong> Distributed mobile NPUs — low per-device efficiency
            </div>
        </div>
    </div>
    """)
    return


# ═══════════════════════════════════════════════════════════════════════════════
# ACT I — THE COMMUNICATION COST REVELATION
# ═══════════════════════════════════════════════════════════════════════════════

@app.cell(hide_code=True)
def _(mo):
    mo.Html("""
    <div style="margin: 28px 0 8px 0;">
        <div style="font-size: 0.72rem; font-weight: 700; letter-spacing: 0.14em;
                    text-transform: uppercase; color: #94a3b8; margin-bottom: 4px;
                    display: flex; align-items: center; gap: 8px;">
            <span style="background: #006395; color: white; border-radius: 50%;
                         width: 20px; height: 20px; display: inline-flex;
                         align-items: center; justify-content: center;
                         font-size: 0.72rem; font-weight: 800; flex-shrink: 0;">I</span>
            Act I · 12–15 min
            <span style="flex: 1; height: 1px; background: #e2e8f0;"></span>
        </div>
        <div style="font-size: 1.55rem; font-weight: 800; color: #0f172a;">
            The Communication Cost Revelation
        </div>
        <div style="font-size: 0.92rem; color: #475569; margin-top: 4px;">
            Federated learning keeps data local. But model gradients are not free.
        </div>
    </div>
    """)
    return


# ─── CELL 5: ACT I STAKEHOLDER MESSAGE (hide_code=True) ──────────────────────
@app.cell(hide_code=True)
def _(mo, COLORS):
    _color = COLORS["Mobile"]
    mo.Html(f"""
    <div style="border-left: 4px solid {_color}; background: #fff7ed;
                border-radius: 0 10px 10px 0; padding: 16px 22px; margin: 12px 0;">
        <div style="font-size: 0.72rem; font-weight: 700; color: {_color};
                    text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px;">
            Incoming Message · Privacy Architect, Android Keyboard Team
        </div>
        <div style="font-style: italic; font-size: 1.0rem; color: #1e293b; line-height: 1.65;">
            "We're training a next-word prediction model on 100M Android devices using
            federated learning. Each device has our 1B-parameter model (2 GB in FP16).
            Per round, each device uploads its full gradient (2 GB). But only 1% of
            devices participate per round — that's still 1M devices. My CTO is asking:
            is the communication cost actually better than just sending user text to
            a central server for training? I need numbers, not marketing."
        </div>
    </div>
    """)
    return


# ─── CELL 6: ACT I CONCEPT SETUP (hide_code=True) ────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    The **federated communication invariant** from @sec-edge-intelligence-federated-systems:

    > Communication cost per round = `model_size_GB × participating_devices × 2`
    > (factor of 2: upload gradient + download updated model)

    The **centralized baseline** from the chapter: users generate ~100 bytes/keystroke.
    At 1B keystrokes/day across 100M users, centralized data upload is ~100 GB/day.

    With 1M devices uploading a 2 GB gradient each round: that is **4 PB per round** —
    before any compression. The question is how many rounds per day the system runs.
    """)
    return


# ─── CELL 7: ACT I PREDICTION LOCK (hide_code=True) ──────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("#### Your Prediction")
    return


@app.cell(hide_code=True)
def _(mo):
    act1_pred = mo.ui.radio(
        options={
            "A) Federated always uses less bandwidth — privacy comes for free": "option_a",
            "B) They are roughly equivalent in bandwidth — federated is a wash": "option_b",
            "C) Without gradient compression, federated uses ~1000x MORE bandwidth than centralized": "option_c",
            "D) Federated uses 10x less bandwidth — keeping data local saves network cost": "option_d",
        },
        label="Compared to centralized training (uploading raw keystrokes), uncompressed federated learning bandwidth is:",
    )
    act1_pred
    return (act1_pred,)


@app.cell(hide_code=True)
def _(mo, act1_pred):
    mo.stop(
        act1_pred.value is None,
        mo.callout(mo.md("Select your prediction to unlock the Act I instruments."), kind="warn")
    )
    mo.callout(mo.md(f"**Prediction locked:** `{act1_pred.value}` — now run the simulator to test it."), kind="info")
    return


# ─── CELL 8: ACT I INSTRUMENTS (hide_code=True) ──────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("#### Act I Instruments — Federated Communication Calculator")
    return


@app.cell(hide_code=True)
def _(mo):
    a1_device_pct = mo.ui.slider(
        start=0.1, stop=10.0, value=1.0, step=0.1,
        label="Participating device fraction (%)",
    )
    a1_model_gb = mo.ui.slider(
        start=0.1, stop=10.0, value=2.0, step=0.1,
        label="Model size per device (GB, FP16 gradients)",
    )
    a1_compression = mo.ui.slider(
        start=1, stop=100, value=1, step=1,
        label="Gradient compression ratio (1x = no compression, 100x = top-K INT8)",
    )
    a1_rounds_per_day = mo.ui.slider(
        start=1, stop=100, value=10, step=1,
        label="Federated rounds per day",
    )
    mo.vstack([
        mo.hstack([a1_device_pct, a1_model_gb], justify="start", gap="2rem"),
        mo.hstack([a1_compression, a1_rounds_per_day], justify="start", gap="2rem"),
    ])
    return (a1_device_pct, a1_model_gb, a1_compression, a1_rounds_per_day)


# ─── CELL 9: ACT I PHYSICS ENGINE (hide_code=True) ───────────────────────────
@app.cell(hide_code=True)
def _(mo, COLORS, a1_device_pct, a1_model_gb, a1_compression, a1_rounds_per_day,
      KEYSTROKES_PER_DAY_B, BYTES_PER_KEYSTROKE):

    # ── Physics: Centralized baseline ────────────────────────────────────────
    # From @sec-edge-intelligence-motivations-benefits-37c3 narrative:
    # 100 bytes/keystroke × 1B keystrokes/day = 100 GB/day raw data upload
    _central_data_gb_per_day = (KEYSTROKES_PER_DAY_B * 1e9 * BYTES_PER_KEYSTROKE) / (1024**3)

    # ── Physics: Federated communication cost ────────────────────────────────
    # From @sec-edge-intelligence-network-bandwidth-optimization-53da:
    # Total devices: 100M
    # Participating per round: fraction% × 100M
    # Upload: model_size_GB per device
    # Download: model_size_GB per device (updated global model)
    # Communication cost per round = participating_devices × model_size × 2 (up+down)
    _total_devices       = 100e6          # 100M Android devices
    _participating       = _total_devices * (a1_device_pct.value / 100.0)
    _model_gb_compressed = a1_model_gb.value / a1_compression.value
    _cost_per_round_gb   = _participating * _model_gb_compressed * 2.0   # upload + download
    _cost_per_round_tb   = _cost_per_round_gb / 1024.0
    _cost_per_round_pb   = _cost_per_round_tb / 1024.0

    # ── Daily federated bandwidth ─────────────────────────────────────────────
    _daily_fed_gb        = _cost_per_round_gb * a1_rounds_per_day.value
    _daily_fed_tb        = _daily_fed_gb / 1024.0
    _daily_fed_pb        = _daily_fed_tb / 1024.0

    # ── Ratio: federated vs centralized ─────────────────────────────────────
    _ratio = _daily_fed_gb / _central_data_gb_per_day  if _central_data_gb_per_day > 0 else 0

    # ── Color coding ─────────────────────────────────────────────────────────
    _ratio_color = (
        COLORS["GreenLine"]  if _ratio < 10   else
        COLORS["OrangeLine"] if _ratio < 100  else
        COLORS["RedLine"]
    )
    _pb_color = (
        COLORS["GreenLine"]  if _daily_fed_pb < 0.1  else
        COLORS["OrangeLine"] if _daily_fed_pb < 1.0  else
        COLORS["RedLine"]
    )

    # ── Format helper ─────────────────────────────────────────────────────────
    def _fmt_bw(gb):
        if gb >= 1024**2:
            return f"{gb/1024**2:.2f} PB"
        elif gb >= 1024:
            return f"{gb/1024:.1f} TB"
        else:
            return f"{gb:.1f} GB"

    mo.md(f"""
    #### Communication Cost Physics

    ```
    Total devices:            100,000,000
    Participating per round:  {_participating:,.0f}  ({a1_device_pct.value:.1f}%)
    Model size (compressed):  {_model_gb_compressed:.2f} GB  ({a1_model_gb.value:.1f} GB ÷ {a1_compression.value}x)
    Cost per round:           {_participating:,.0f} × {_model_gb_compressed:.2f} GB × 2 = {_fmt_bw(_cost_per_round_gb)}
    Rounds per day:           {a1_rounds_per_day.value}
    Daily federated BW:       {_fmt_bw(_daily_fed_gb)}
    Daily centralized BW:     {_fmt_bw(_central_data_gb_per_day)}  (raw keystroke data)
    Federated / Centralized:  {_ratio:.0f}×
    ```

    <div style="display: flex; gap: 20px; justify-content: start; flex-wrap: wrap; margin-top: 20px;">
        <div style="padding: 20px; border: 1px solid #e2e8f0; border-radius: 10px;
                    min-width: 180px; text-align: center; background: white;">
            <div style="color: #64748b; font-size: 0.82rem; margin-bottom: 6px;">Cost Per Round</div>
            <div style="font-size: 1.7rem; font-weight: 800; color: {_pb_color};">
                {_fmt_bw(_cost_per_round_gb)}
            </div>
        </div>
        <div style="padding: 20px; border: 1px solid #e2e8f0; border-radius: 10px;
                    min-width: 180px; text-align: center; background: white;">
            <div style="color: #64748b; font-size: 0.82rem; margin-bottom: 6px;">Daily Federated BW</div>
            <div style="font-size: 1.7rem; font-weight: 800; color: {_pb_color};">
                {_fmt_bw(_daily_fed_gb)}
            </div>
        </div>
        <div style="padding: 20px; border: 1px solid #e2e8f0; border-radius: 10px;
                    min-width: 180px; text-align: center; background: white;">
            <div style="color: #64748b; font-size: 0.82rem; margin-bottom: 6px;">Daily Centralized BW</div>
            <div style="font-size: 1.7rem; font-weight: 800; color: #008F45;">
                {_fmt_bw(_central_data_gb_per_day)}
            </div>
        </div>
        <div style="padding: 20px; border: 1px solid #e2e8f0; border-radius: 10px;
                    min-width: 180px; text-align: center; background: white;">
            <div style="color: #64748b; font-size: 0.82rem; margin-bottom: 6px;">Fed / Centralized</div>
            <div style="font-size: 1.7rem; font-weight: 800; color: {_ratio_color};">
                {_ratio:.0f}×
            </div>
        </div>
    </div>
    """)
    return (
        _central_data_gb_per_day,
        _daily_fed_gb,
        _ratio,
        _participating,
        _model_gb_compressed,
        _cost_per_round_gb,
        _daily_fed_pb,
    )


# ─── CELL 10: ACT I BANDWIDTH CHART (hide_code=True) ─────────────────────────
@app.cell(hide_code=True)
def _(mo, go, np, COLORS, apply_plotly_theme,
      a1_model_gb, a1_compression, a1_rounds_per_day,
      _central_data_gb_per_day):

    # ── Sweep participation rate 0.1% → 10% ──────────────────────────────────
    _pct_range    = np.linspace(0.1, 10.0, 50)
    _total_dev    = 100e6
    _mdl_compressed = a1_model_gb.value / a1_compression.value

    _daily_fed_curve = (
        (_pct_range / 100.0) * _total_dev
        * _mdl_compressed * 2.0
        * a1_rounds_per_day.value
        / 1024.0  # → TB
    )
    _central_tb = _central_data_gb_per_day / 1024.0

    fig_act1 = go.Figure()

    # Centralized baseline
    fig_act1.add_trace(go.Scatter(
        x=_pct_range,
        y=[_central_tb] * len(_pct_range),
        mode="lines",
        name="Centralized (raw data)",
        line=dict(color=COLORS["Cloud"], width=2, dash="dash"),
    ))

    # Federated curve
    fig_act1.add_trace(go.Scatter(
        x=_pct_range,
        y=_daily_fed_curve,
        mode="lines",
        name="Federated (gradient upload)",
        line=dict(color=COLORS["Mobile"], width=2.5),
        fill="tozeroy",
        fillcolor="rgba(204,85,0,0.08)",
    ))

    fig_act1.update_layout(
        title="Daily Bandwidth vs Device Participation Rate",
        xaxis_title="Participating Devices (%)",
        yaxis_title="Daily Bandwidth (TB)",
        legend=dict(x=0.02, y=0.98),
        height=320,
        yaxis_type="log",
    )
    apply_plotly_theme(fig_act1)
    mo.ui.plotly(fig_act1)
    return (fig_act1,)


# ─── CELL 11: ACT I PREDICTION OVERLAY (hide_code=True) ──────────────────────
@app.cell(hide_code=True)
def _(mo, act1_pred, _ratio):
    _pred_map = {
        "option_a": 0.1,    # "always less" — implies < 1x
        "option_b": 1.0,    # "roughly equivalent" — implies ~1x
        "option_c": 1000.0, # correct answer — ~1000x
        "option_d": 0.1,    # "10x less" — implies < 1x
    }
    _pred_val  = _pred_map.get(act1_pred.value, 1.0)
    _actual    = _ratio
    _gap       = abs(_actual - _pred_val) / max(_pred_val, 1.0)
    _is_close  = _gap < 0.5

    mo.callout(mo.md(
        f"**You predicted:** federated bandwidth ratio ≈ `{_pred_val:.0f}×` centralized.\n\n"
        f"**The simulator shows:** `{_actual:.0f}×` (at current settings).\n\n"
        f"{'**Close call.** Your intuition was well-calibrated for these parameters.' if _is_close else '**Significant gap.** The physics diverged from intuition — this is where learning happens.'} "
        f"At 1% participation with 2 GB model and no compression: "
        f"1M devices × 2 GB × 2 (up+down) = **4 PB/round**. "
        f"Centralized baseline is only ~100 GB/day. "
        f"Uncompressed federated uses **orders of magnitude more bandwidth** than sending raw keystrokes."
    ), kind="success" if _is_close else "warn")
    return


# ─── CELL 12: ACT I REFLECTION (hide_code=True) ──────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("#### Reflection: Production Federated Learning (Gboard)")
    return


@app.cell(hide_code=True)
def _(mo):
    act1_reflect = mo.ui.radio(
        options={
            "A) Reduce the participating device count — fewer devices = less bandwidth": "reflect_a",
            "B) Top-K gradient sparsification + quantization — only upload the 1% largest gradient values in INT8": "reflect_b",
            "C) Send only loss values, not gradients — the server can infer weights from loss": "reflect_c",
            "D) Reduce model size aggressively — smaller model means smaller upload": "reflect_d",
        },
        label="In production federated learning (e.g., Google Gboard), what is the PRIMARY bandwidth reduction technique?",
    )
    act1_reflect
    return (act1_reflect,)


@app.cell(hide_code=True)
def _(mo, act1_reflect):
    mo.stop(
        act1_reflect.value is None,
        mo.callout(mo.md("Select your answer to see the explanation."), kind="warn")
    )

    _correct = act1_reflect.value == "reflect_b"
    _feedback = {
        "reflect_a": mo.callout(mo.md(
            "**Incorrect.** Reducing participation improves privacy diversity "
            "coverage but does not address the per-device bandwidth. Worse, "
            "fewer participants degrade model quality. The communication problem "
            "is not about *how many* devices — it is about *how much data per device*."
        ), kind="warn"),
        "reflect_b": mo.callout(mo.md(
            "**Correct.** Google Gboard uses **top-K gradient sparsification** combined "
            "with **INT8 quantization** — transmitting only the 1% largest gradient values "
            "in 8-bit integers rather than FP32. From @sec-edge-intelligence-network-bandwidth-optimization-53da: "
            "*'Gradient quantization reduces precision from FP32 to INT8 or even binary representations, "
            "achieving 4–32× compression with minimal accuracy loss. Top-K gradient selection further reduces "
            "communication by transmitting only the most significant parameter updates.'* "
            "Combined: 100× compression ratio, bringing 4 PB/round down to ~40 TB/round. "
            "Error accumulation ensures small gradients are not permanently lost."
        ), kind="success"),
        "reflect_c": mo.callout(mo.md(
            "**Incorrect.** The server cannot reconstruct gradient updates from loss values alone — "
            "loss is a scalar that collapses all gradient information. The server needs gradients "
            "(or model weight deltas) to perform FedAvg. Sending only loss values would make "
            "federated learning impossible."
        ), kind="warn"),
        "reflect_d": mo.callout(mo.md(
            "**Incorrect.** While smaller models help, reducing model size hurts prediction quality — "
            "the keyboard suggestion use case requires a 1B+ parameter model for acceptable accuracy. "
            "Production systems solve bandwidth via *compression of the existing model's gradients*, "
            "not by shrinking the model. Both dimensions (compression ratio and model size) matter, "
            "but compression is the primary lever."
        ), kind="warn"),
    }
    _feedback.get(act1_reflect.value, mo.callout(mo.md("Select an option."), kind="info"))
    return


# ─── CELL 13: ACT I MATHPEEK (hide_code=True) ────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "The governing equation — Federated Communication Cost": mo.md("""
**Federated Communication Cost Per Round**

$$C_{round} = N_{participating} \\times M_{compressed} \\times 2$$

Where:
- **$N_{participating}$** — number of devices in this round = $N_{total} \\times f_{participation}$
- **$M_{compressed}$** — compressed gradient size per device (GB) = $M_{model} \\div r_{compression}$
- **Factor of 2** — upload gradient (device → server) + download updated model (server → device)

**Daily bandwidth:**

$$C_{daily} = C_{round} \\times R_{rounds/day}$$

**Gradient compression analysis (from @sec-edge-intelligence-network-bandwidth-optimization-53da):**

Top-K sparsification + INT8 quantization achieves:
$$r_{compression} = \\underbrace{100}_{\\text{top-K}} \\times \\underbrace{4}_{\\text{FP32→INT8}} = 400\\times$$

In practice, Gboard achieves ~100× compression with error accumulation to prevent gradient loss.

**Centralized baseline:**

$$C_{centralized} = N_{keystrokes/day} \\times B_{keystroke} = 10^9 \\times 100 \\text{ bytes} = 100 \\text{ GB/day}$$

**Ratio at 1% participation, 2 GB model, no compression, 10 rounds/day:**

$$\\frac{C_{federated}}{C_{centralized}} = \\frac{10^6 \\times 2 \\text{ GB} \\times 2 \\times 10}{0.1 \\text{ TB}} = \\frac{40 \\text{ PB}}{0.1 \\text{ TB}} \\approx 400{,}000\\times$$

This is the federation paradox: the privacy-preserving approach uses *more* bandwidth than sending raw data, not less.
        """),
    })
    return


# ═══════════════════════════════════════════════════════════════════════════════
# ACT II — THE PRIVACY-UTILITY TRADEOFF
# ═══════════════════════════════════════════════════════════════════════════════

@app.cell(hide_code=True)
def _(mo):
    mo.Html("""
    <div style="margin: 36px 0 8px 0;">
        <div style="font-size: 0.72rem; font-weight: 700; letter-spacing: 0.14em;
                    text-transform: uppercase; color: #94a3b8; margin-bottom: 4px;
                    display: flex; align-items: center; gap: 8px;">
            <span style="background: #CC5500; color: white; border-radius: 50%;
                         width: 20px; height: 20px; display: inline-flex;
                         align-items: center; justify-content: center;
                         font-size: 0.72rem; font-weight: 800; flex-shrink: 0;">II</span>
            Act II · 20–25 min
            <span style="flex: 1; height: 1px; background: #e2e8f0;"></span>
        </div>
        <div style="font-size: 1.55rem; font-weight: 800; color: #0f172a;">
            The Privacy-Utility Tradeoff
        </div>
        <div style="font-size: 0.92rem; color: #475569; margin-top: 4px;">
            Differential privacy provides formal guarantees — but at an accuracy cost.
            Design the system that survives both constraints.
        </div>
    </div>
    """)
    return


# ─── CELL 14: ACT II STAKEHOLDER MESSAGE (hide_code=True) ────────────────────
@app.cell(hide_code=True)
def _(mo, COLORS):
    _color = COLORS["Mobile"]
    mo.Html(f"""
    <div style="border-left: 4px solid {_color}; background: #fff7ed;
                border-radius: 0 10px 10px 0; padding: 16px 22px; margin: 12px 0;">
        <div style="font-size: 0.72rem; font-weight: 700; color: {_color};
                    text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px;">
            Incoming Message · Product Lead, Personalization Platform
        </div>
        <div style="font-style: italic; font-size: 1.0rem; color: #1e293b; line-height: 1.65;">
            "We have three deployment options for our on-device recommendation model:
            (A) Centralized cloud training — best accuracy, worst privacy.
            (B) Federated learning without differential privacy — good privacy story, but
            model inversion attacks are still possible.
            (C) Federated learning with ε=1 differential privacy — formal mathematical guarantee.
            User survey: 73% prefer option C. But our ML team says accuracy drops 8% vs centralized.
            My engineering question is: which option actually provides a *formal* privacy guarantee,
            and can we find ε that keeps accuracy within 5% of centralized while staying private?"
        </div>
    </div>
    """)
    return


# ─── CELL 15: ACT II CONCEPT SETUP (hide_code=True) ──────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    From @sec-edge-intelligence-federated-privacy-a1ed, the privacy landscape:

    - **Option A (Centralized):** All raw user data on server. No privacy guarantee.
      Accuracy ceiling = 100% (baseline).
    - **Option B (Federated, no DP):** Data stays local. But gradient inversion attacks can
      reconstruct training samples from gradients. *Not formally private.*
    - **Option C (Federated + DP):** Gaussian noise added to gradients before upload.
      Provides ε-δ differential privacy. *Only option with a mathematical guarantee.*

    The DP-SGD noise mechanism: `σ = C · sqrt(2 · ln(1.25/δ)) / ε`

    Smaller ε = stronger privacy = more noise = lower accuracy.
    The design challenge: find ε where utility loss is acceptable.
    """)
    return


# ─── CELL 16: ACT II PREDICTION LOCK (hide_code=True) ────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("#### Your Prediction")
    return


@app.cell(hide_code=True)
def _(mo):
    act2_pred = mo.ui.radio(
        options={
            "A) Option A — centralized training; accuracy always trumps privacy for recommendation": "pred2_a",
            "B) Option B — federated without DP gives strong practical privacy anyway": "pred2_b",
            "C) Option C — differential privacy (ε=1) is the only option providing formal mathematical guarantees": "pred2_c",
            "D) All three options are equivalent in practice — theoretical distinctions don't matter": "pred2_d",
        },
        label="Which deployment option provides a *formal mathematical privacy guarantee*?",
    )
    act2_pred
    return (act2_pred,)


@app.cell(hide_code=True)
def _(mo, act2_pred):
    mo.stop(
        act2_pred.value is None,
        mo.callout(mo.md("Select your prediction to unlock the Act II design instruments."), kind="warn")
    )
    mo.callout(mo.md(f"**Prediction locked:** `{act2_pred.value}` — configure the system below."), kind="info")
    return


# ─── CELL 17: ACT II INSTRUMENTS (hide_code=True) ────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("#### Act II Instruments — Federated Learning Designer")
    return


@app.cell(hide_code=True)
def _(mo):
    a2_epsilon = mo.ui.slider(
        start=0.1, stop=10.0, value=1.0, step=0.1,
        label="DP privacy budget ε (smaller = stronger privacy = more noise)",
    )
    a2_local_epochs = mo.ui.slider(
        start=1, stop=20, value=5, step=1,
        label="Local SGD epochs per round (more epochs → better local convergence)",
    )
    a2_part_frac = mo.ui.slider(
        start=0.1, stop=10.0, value=1.0, step=0.1,
        label="Participating device fraction (%)",
    )
    a2_agg_rounds = mo.ui.slider(
        start=10, stop=500, value=100, step=10,
        label="Aggregation rounds (total training rounds)",
    )
    mo.vstack([
        mo.hstack([a2_epsilon, a2_local_epochs], justify="start", gap="2rem"),
        mo.hstack([a2_part_frac, a2_agg_rounds], justify="start", gap="2rem"),
    ])
    return (a2_epsilon, a2_local_epochs, a2_part_frac, a2_agg_rounds)


# ─── CELL 18: ACT II PHYSICS ENGINE (hide_code=True) ─────────────────────────
@app.cell(hide_code=True)
def _(mo, COLORS,
      a2_epsilon, a2_local_epochs, a2_part_frac, a2_agg_rounds,
      MOBILE_RAM_GB):

    # ── Physics: DP-SGD noise magnitude ──────────────────────────────────────
    # From @sec-edge-intelligence-federated-privacy-a1ed:
    # Gaussian mechanism: σ = C * sqrt(2 * ln(1.25/δ)) / ε
    # Standard settings: gradient clipping C = 1.0, δ = 1e-5
    import math as _math

    _eps    = a2_epsilon.value
    _delta  = 1e-5          # standard δ in DP-SGD literature
    _clip_C = 1.0           # gradient clipping norm
    _sigma  = _clip_C * _math.sqrt(2.0 * _math.log(1.25 / _delta)) / _eps

    # ── Physics: Accuracy vs epsilon curve ───────────────────────────────────
    # Based on chapter claim: ε=1 DP causes ~8% accuracy drop vs centralized
    # Model: accuracy_drop = k / (1 + ε) where k is calibrated to 8% at ε=1
    # Source: chapter narrative on Option C, 8% drop claim
    _ACC_CENTRALIZED  = 92.0   # % — typical recommendation model accuracy (baseline)
    _K_DP_ACCURACY    = 8.0    # calibration constant: 8% drop at ε=1
    _acc_drop_dp      = _K_DP_ACCURACY / (1.0 + _eps)  # decreasing as ε increases
    _acc_federated_dp = _ACC_CENTRALIZED - _acc_drop_dp

    # Local epochs effect: more local epochs reduce rounds needed but increase drift
    # Convergence scaling: effective_rounds = agg_rounds × local_epochs
    _effective_compute = a2_agg_rounds.value * a2_local_epochs.value
    # Accuracy bonus from more local epochs (up to a ceiling from non-IID drift)
    _epoch_bonus = min(2.0, a2_local_epochs.value * 0.15)
    _acc_final   = _acc_federated_dp + _epoch_bonus

    # Cap at centralized accuracy
    _acc_final = min(_acc_final, _ACC_CENTRALIZED)

    # Accuracy gap vs centralized
    _acc_gap    = _ACC_CENTRALIZED - _acc_final
    _utility_ok = _acc_gap <= 5.0   # within 5% of centralized = acceptable

    # ── Physics: Communication cost ───────────────────────────────────────────
    # Model size for recommendation: 100M params, FP16 = 200 MB = 0.2 GB
    # With DP noise, gradient size unchanged (noise added before transmission)
    _rec_model_gb     = 0.2    # 100M param recommendation model in FP16
    _total_dev        = 100e6
    _participating    = _total_dev * (a2_part_frac.value / 100.0)
    _cost_round_gb    = _participating * _rec_model_gb * 2.0
    _daily_bw_gb      = _cost_round_gb * (a2_agg_rounds.value / 30.0)  # assume ~30 days
    _daily_bw_tb      = _daily_bw_gb / 1024.0
    _daily_bw_pb      = _daily_bw_tb / 1024.0
    _bw_ok            = _daily_bw_pb < 1.0

    # ── Privacy guarantee label ────────────────────────────────────────────────
    _privacy_str = f"ε={_eps:.1f} DP (σ={_sigma:.2f})"

    # ── Color coding ──────────────────────────────────────────────────────────
    _acc_color = (
        COLORS["GreenLine"]  if _acc_gap <= 5.0   else
        COLORS["OrangeLine"] if _acc_gap <= 10.0  else
        COLORS["RedLine"]
    )
    _sigma_color = (
        COLORS["GreenLine"]  if _sigma <= 1.0  else
        COLORS["OrangeLine"] if _sigma <= 3.0  else
        COLORS["RedLine"]
    )
    _bw_color = (
        COLORS["GreenLine"]  if _daily_bw_pb < 0.5  else
        COLORS["OrangeLine"] if _daily_bw_pb < 1.0  else
        COLORS["RedLine"]
    )

    mo.md(f"""
    #### DP-SGD Physics

    ```
    Privacy budget ε:         {_eps:.1f}
    DP noise magnitude σ:     {_sigma:.3f}  (σ = C·√(2·ln(1.25/δ)) / ε)
    Local epochs per round:   {a2_local_epochs.value}
    Aggregation rounds:       {a2_agg_rounds.value}
    Effective compute:        {_effective_compute:,} (rounds × epochs)

    Centralized accuracy:     {_ACC_CENTRALIZED:.1f}%
    DP accuracy:              {_acc_final:.1f}%
    Accuracy gap:             {_acc_gap:.1f}%  (target: ≤ 5%)
    Within utility threshold: {'YES' if _utility_ok else 'NO — MODEL UTILITY COMPROMISED'}

    Daily communication BW:   {_daily_bw_pb:.2f} PB  (target: < 1 PB)
    BW constraint satisfied:  {'YES' if _bw_ok else 'NO — BANDWIDTH BUDGET EXCEEDED'}
    ```

    <div style="display: flex; gap: 20px; justify-content: start; flex-wrap: wrap; margin-top: 20px;">
        <div style="padding: 20px; border: 1px solid #e2e8f0; border-radius: 10px;
                    min-width: 180px; text-align: center; background: white;">
            <div style="color: #64748b; font-size: 0.82rem; margin-bottom: 6px;">DP Noise σ</div>
            <div style="font-size: 1.7rem; font-weight: 800; color: {_sigma_color};">
                {_sigma:.2f}
            </div>
        </div>
        <div style="padding: 20px; border: 1px solid #e2e8f0; border-radius: 10px;
                    min-width: 180px; text-align: center; background: white;">
            <div style="color: #64748b; font-size: 0.82rem; margin-bottom: 6px;">Model Accuracy</div>
            <div style="font-size: 1.7rem; font-weight: 800; color: {_acc_color};">
                {_acc_final:.1f}%
            </div>
        </div>
        <div style="padding: 20px; border: 1px solid #e2e8f0; border-radius: 10px;
                    min-width: 180px; text-align: center; background: white;">
            <div style="color: #64748b; font-size: 0.82rem; margin-bottom: 6px;">Accuracy Gap</div>
            <div style="font-size: 1.7rem; font-weight: 800; color: {_acc_color};">
                {_acc_gap:.1f}%
            </div>
        </div>
        <div style="padding: 20px; border: 1px solid #e2e8f0; border-radius: 10px;
                    min-width: 180px; text-align: center; background: white;">
            <div style="color: #64748b; font-size: 0.82rem; margin-bottom: 6px;">Daily BW</div>
            <div style="font-size: 1.7rem; font-weight: 800; color: {_bw_color};">
                {_daily_bw_pb:.2f} PB
            </div>
        </div>
    </div>
    """)

    return (
        _eps,
        _sigma,
        _acc_final,
        _acc_gap,
        _utility_ok,
        _daily_bw_pb,
        _bw_ok,
        _privacy_str,
        _ACC_CENTRALIZED,
    )


# ─── CELL 19: ACT II FAILURE STATES (hide_code=True) ─────────────────────────
@app.cell(hide_code=True)
def _(mo, _acc_gap, _daily_bw_pb, _eps, _utility_ok, _bw_ok):

    _items = []

    # Failure 1: Model utility collapse (kind="danger")
    if not _utility_ok:
        _items.append(mo.callout(mo.md(
            f"**Model utility collapse:** DP with ε={_eps:.1f} causes {_acc_gap:.1f}% accuracy "
            f"loss vs centralized — exceeds the 5% utility threshold. "
            f"**Fix:** Increase ε (weakens privacy but restores accuracy), "
            f"increase local epochs to improve gradient signal-to-noise ratio, "
            f"or switch to DP-SGD with adaptive clipping to reduce noise magnitude."
        ), kind="danger"))

    # Failure 2: Communication budget exceeded (kind="warn")
    if not _bw_ok:
        _items.append(mo.callout(mo.md(
            f"**Communication budget exceeded.** Daily bandwidth: {_daily_bw_pb:.1f} PB. "
            f"Budget target: < 1 PB/day. "
            f"**Fix:** Reduce participating fraction, apply gradient compression (100× ratio "
            f"from top-K INT8 sparsification), or reduce aggregation round frequency."
        ), kind="warn"))

    # Success state
    if _utility_ok and _bw_ok:
        _items.append(mo.callout(mo.md(
            f"**Feasible design found.** ε={_eps:.1f} provides formal DP guarantee "
            f"with only {_acc_gap:.1f}% accuracy loss (within 5% threshold) "
            f"and {_daily_bw_pb:.2f} PB/day bandwidth (under 1 PB budget). "
            f"This configuration is deployable."
        ), kind="success"))

    mo.vstack(_items) if _items else mo.md("")
    return


# ─── CELL 20: ACT II ACCURACY vs EPSILON CHART (hide_code=True) ──────────────
@app.cell(hide_code=True)
def _(mo, go, np, COLORS, apply_plotly_theme, a2_epsilon, _ACC_CENTRALIZED):

    # ── Sweep ε from 0.1 to 10 ───────────────────────────────────────────────
    _eps_range   = np.linspace(0.1, 10.0, 100)
    _K_dp        = 8.0
    _epoch_bonus = 1.5   # fixed for chart (5 local epochs)
    _acc_dp_curve = np.minimum(
        _ACC_CENTRALIZED,
        _ACC_CENTRALIZED - _K_dp / (1.0 + _eps_range) + _epoch_bonus
    )
    _acc_dp_curve = np.maximum(_acc_dp_curve, 0.0)

    _acc_centralized_line = np.full_like(_eps_range, _ACC_CENTRALIZED)
    _threshold_line       = np.full_like(_eps_range, _ACC_CENTRALIZED - 5.0)  # 5% gap threshold

    fig_act2 = go.Figure()

    # Centralized ceiling
    fig_act2.add_trace(go.Scatter(
        x=_eps_range,
        y=_acc_centralized_line,
        mode="lines",
        name="Centralized (no DP)",
        line=dict(color=COLORS["Cloud"], width=2, dash="dash"),
    ))

    # 5% utility threshold
    fig_act2.add_trace(go.Scatter(
        x=_eps_range,
        y=_threshold_line,
        mode="lines",
        name="5% utility threshold",
        line=dict(color=COLORS["OrangeLine"], width=1.5, dash="dot"),
    ))

    # DP accuracy curve
    fig_act2.add_trace(go.Scatter(
        x=_eps_range,
        y=_acc_dp_curve,
        mode="lines",
        name="Federated + DP accuracy",
        line=dict(color=COLORS["Mobile"], width=2.5),
        fill="tozeroy",
        fillcolor="rgba(204,85,0,0.07)",
    ))

    # Current operating point
    _cur_eps    = a2_epsilon.value
    _cur_acc    = float(np.minimum(
        _ACC_CENTRALIZED,
        _ACC_CENTRALIZED - _K_dp / (1.0 + _cur_eps) + _epoch_bonus
    ))
    fig_act2.add_trace(go.Scatter(
        x=[_cur_eps],
        y=[_cur_acc],
        mode="markers",
        name="Current ε",
        marker=dict(color=COLORS["RedLine"], size=12, symbol="diamond",
                    line=dict(color="white", width=2)),
    ))

    fig_act2.update_layout(
        title="Model Accuracy vs Privacy Budget ε",
        xaxis_title="Privacy Budget ε (higher = less private)",
        yaxis_title="Model Accuracy (%)",
        legend=dict(x=0.02, y=0.15),
        height=340,
        yaxis=dict(range=[78, 95]),
    )
    apply_plotly_theme(fig_act2)
    mo.ui.plotly(fig_act2)
    return (fig_act2,)


# ─── CELL 21: ACT II PREDICTION REVEAL (hide_code=True) ──────────────────────
@app.cell(hide_code=True)
def _(mo, act2_pred, _acc_gap, _eps):
    _correct2 = act2_pred.value == "pred2_c"

    _feedback2 = {
        "pred2_a": mo.callout(mo.md(
            "**Incorrect.** Option A (centralized) provides no formal privacy guarantee — "
            "all raw user data is transmitted to and stored on the server. It achieves "
            "the best accuracy but the worst privacy posture. The 73% of users who prefer "
            "Option C have the right intuition."
        ), kind="warn"),
        "pred2_b": mo.callout(mo.md(
            "**Incorrect.** Federated learning without differential privacy does *not* "
            "provide formal privacy guarantees. From @sec-edge-intelligence-federated-privacy-a1ed: "
            "*'Although devices do not share their raw data, the transmitted model updates "
            "can inadvertently leak information... Model inversion attacks and membership "
            "inference attacks demonstrate that adversaries may partially reconstruct or "
            "infer properties of local datasets by analyzing these updates.'* "
            "Federated without DP is a *practical* privacy improvement but not a *mathematical* one."
        ), kind="warn"),
        "pred2_c": mo.callout(mo.md(
            f"**Correct.** Option C — Federated Learning with ε-δ Differential Privacy — "
            f"is the **only option providing a formal mathematical guarantee**. "
            f"DP guarantees that an adversary observing model outputs cannot distinguish "
            f"whether any individual's data was included, with probability bounded by e^ε. "
            f"At ε={_eps:.1f}, the accuracy cost is {_acc_gap:.1f}% vs centralized. "
            f"The engineering challenge is finding ε where this cost is acceptable — "
            f"exactly what Act II instruments let you explore."
        ), kind="success"),
        "pred2_d": mo.callout(mo.md(
            "**Incorrect.** The three options have fundamentally different privacy properties. "
            "Option A: no privacy. Option B: practical but informal privacy. "
            "Option C: formal mathematical guarantee via differential privacy. "
            "These distinctions matter in regulated industries (healthcare, finance) where "
            "'we use federated learning' is not a sufficient compliance argument — "
            "only formal ε-δ DP satisfies legal standards like GDPR Article 89."
        ), kind="warn"),
    }
    _feedback2.get(act2_pred.value, mo.callout(mo.md("Select an option."), kind="info"))
    return


# ─── CELL 22: ACT II REFLECTION (hide_code=True) ─────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("#### Reflection: What Does ε=1 Actually Mean?")
    return


@app.cell(hide_code=True)
def _(mo):
    act2_reflect = mo.ui.radio(
        options={
            "A) Only 1% of users' data is protected — 99% can leak": "ref2_a",
            "B) Adding or removing one user's data changes any output probability by at most e^1 ≈ 2.7× — the privacy-utility parameter": "ref2_b",
            "C) Exactly 1 bit of information leaks per query to the model": "ref2_c",
            "D) The privacy budget expires after 1 training round — ε resets per round": "ref2_d",
        },
        label="What does ε=1 in differential privacy mean practically?",
    )
    act2_reflect
    return (act2_reflect,)


@app.cell(hide_code=True)
def _(mo, act2_reflect):
    mo.stop(
        act2_reflect.value is None,
        mo.callout(mo.md("Select your answer to see the explanation."), kind="warn")
    )

    _reflect2_feedback = {
        "ref2_a": mo.callout(mo.md(
            "**Incorrect.** ε=1 does not mean 1% of data is protected. "
            "Differential privacy is a property of the *algorithm*, not a fraction of the dataset. "
            "With ε-DP, *all* users receive privacy protection simultaneously — "
            "the ε parameter controls the *strength* of that protection, not its coverage."
        ), kind="warn"),
        "ref2_b": mo.callout(mo.md(
            "**Correct.** The formal definition of ε-differential privacy: for any output S "
            "of a mechanism M, `P[M(D) ∈ S] ≤ e^ε × P[M(D') ∈ S]` for any two datasets D, D' "
            "that differ by one record. At ε=1: `e^1 ≈ 2.718`. This means an adversary "
            "observing any output cannot distinguish by more than 2.7× whether your specific "
            "data was included. Smaller ε → tighter bound → stronger privacy. "
            "The tradeoff: smaller ε requires larger Gaussian noise σ (σ = C·√(2·ln(1.25/δ))/ε), "
            "which degrades model accuracy."
        ), kind="success"),
        "ref2_c": mo.callout(mo.md(
            "**Incorrect.** ε is not measured in bits of information leakage. "
            "It is a bound on the *multiplicative change in output probabilities*. "
            "While there are connections to information-theoretic privacy concepts like "
            "mutual information, ε-DP does not directly correspond to bit-level leakage. "
            "The correct interpretation is the probability ratio bound: e^ε."
        ), kind="warn"),
        "ref2_d": mo.callout(mo.md(
            "**Incorrect.** ε does not reset per round — this is one of the most important "
            "system design implications of DP. Privacy budgets **compose**: running T rounds of "
            "ε-DP training uses O(ε√T) total privacy budget (under advanced composition). "
            "This is why DP-SGD in production uses **privacy accounting** (e.g., Rényi DP) "
            "to track cumulative privacy loss across all training rounds. "
            "A system running 1000 rounds with ε=0.1 per round provides *less* privacy "
            "than one running 100 rounds with ε=1."
        ), kind="warn"),
    }
    _reflect2_feedback.get(act2_reflect.value, mo.callout(mo.md("Select an option."), kind="info"))
    return


# ─── CELL 23: ACT II MATHPEEK (hide_code=True) ───────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "The governing equations — Differential Privacy and Federated Convergence": mo.md("""
**ε-δ Differential Privacy (formal definition)**

A randomized mechanism M satisfies (ε, δ)-DP if for all datasets D, D' differing by one record,
and for all outputs S:

$$P[M(D) \\in S] \\leq e^\\varepsilon \\cdot P[M(D') \\in S] + \\delta$$

- **ε** — privacy budget (smaller = stronger guarantee = more noise)
- **δ** — failure probability (typically 10⁻⁵ — chance that ε-bound is exceeded)

**Gaussian Mechanism noise parameter**

$$\\sigma = \\frac{C \\cdot \\sqrt{2 \\ln(1.25 / \\delta)}}{\\varepsilon}$$

Where C is the gradient clipping norm. At ε=1, δ=10⁻⁵:

$$\\sigma = \\frac{1 \\cdot \\sqrt{2 \\ln(125{,}000)}}{1} = \\frac{1 \\cdot \\sqrt{2 \\times 11.74}}{1} \\approx 4.84$$

**Federated Averaging convergence bound** (from @sec-edge-intelligence-federated-learning-convergence-analysis-c1fc)

$$\\varepsilon_{gap} \\leq \\frac{\\sigma^2}{C \\cdot E \\cdot R} + \\frac{\\beta^2 E^2}{R}$$

Where:
- **C** — clients per round (participating fraction × total)
- **E** — local epochs per round
- **R** — total aggregation rounds
- **β** — data heterogeneity factor (0 = IID, >1 = severe non-IID)
- **σ** — gradient noise (including DP noise)

**Privacy budget composition** (Advanced Composition Theorem):

Running T rounds of ε-DP training incurs total privacy cost:

$$\\varepsilon_{total} \\approx \\varepsilon \\sqrt{T \\cdot \\ln(1/\\delta)}$$

This is why production systems use privacy accounting (Rényi DP moments accountant) to
track cumulative budget and stop training before budget exhaustion.
        """),
    })
    return


# ═══════════════════════════════════════════════════════════════════════════════
# LEDGER SAVE + HUD FOOTER
# ═══════════════════════════════════════════════════════════════════════════════

@app.cell(hide_code=True)
def _(mo, ledger, COLORS,
      context_toggle, act1_pred, act2_pred,
      a2_epsilon, a2_part_frac,
      _daily_fed_pb, _acc_gap, _utility_ok, _privacy_str,
      _ratio, _bw_ok):

    # ── Determine correctness ─────────────────────────────────────────────────
    _act1_correct = (act1_pred.value == "option_c")
    _act2_correct = (act2_pred.value == "pred2_c")
    _constraint_hit = (not _utility_ok) or (not _bw_ok)

    # ── Privacy guarantee string ──────────────────────────────────────────────
    _priv_guarantee = (
        "dp"        if a2_epsilon.value <= 5.0  else
        "federated" if context_toggle.value == "federated" else
        "none"
    )

    # ── Save to ledger ────────────────────────────────────────────────────────
    ledger.save(
        chapter="v2_11",
        design={
            "context":                context_toggle.value,
            "dp_epsilon":             float(a2_epsilon.value),
            "participating_fraction": float(a2_part_frac.value),
            "compression_ratio":      1.0,   # default (Act I compression tracked separately)
            "daily_bandwidth_tb":     float(_daily_fed_pb * 1024.0),
            "accuracy_vs_centralized": float(100.0 - _acc_gap),
            "act1_prediction":        str(act1_pred.value),
            "act1_correct":           bool(_act1_correct),
            "act2_result":            float(_acc_gap),
            "act2_decision":          str(act2_pred.value),
            "constraint_hit":         bool(_constraint_hit),
            "privacy_guarantee":      str(_priv_guarantee),
        }
    )

    # ── HUD footer ────────────────────────────────────────────────────────────
    _track   = ledger.get_track() or "—"
    _ch_str  = "V2-11"
    _ctx_str = context_toggle.value.upper()

    _act1_badge = (
        f'<span class="hud-active">ACT I ✓ correct</span>'
        if _act1_correct else
        f'<span class="hud-none">ACT I ✗ prediction missed</span>'
    )
    _act2_badge = (
        f'<span class="hud-active">ACT II ✓ correct</span>'
        if _act2_correct else
        f'<span class="hud-none">ACT II ✗ prediction missed</span>'
    )
    _constraint_badge = (
        f'<span class="hud-none">CONSTRAINT HIT</span>'
        if _constraint_hit else
        f'<span class="hud-active">CONSTRAINTS OK</span>'
    )

    mo.Html(f"""
    <div class="lab-hud">
        <span><span class="hud-label">LAB</span>&nbsp;
              <span class="hud-value">{_ch_str}</span></span>
        <span><span class="hud-label">TRACK</span>&nbsp;
              <span class="hud-value">{_track}</span></span>
        <span><span class="hud-label">CONTEXT</span>&nbsp;
              <span class="hud-value">{_ctx_str}</span></span>
        <span><span class="hud-label">ε</span>&nbsp;
              <span class="hud-value">{a2_epsilon.value:.1f}</span></span>
        <span><span class="hud-label">ACCURACY GAP</span>&nbsp;
              <span class="hud-value">{_acc_gap:.1f}%</span></span>
        <span>{_act1_badge}</span>
        <span>{_act2_badge}</span>
        <span>{_constraint_badge}</span>
        <span><span class="hud-label">PRIVACY</span>&nbsp;
              <span class="hud-active">{_priv_guarantee.upper()}</span></span>
    </div>
    """)
    return


if __name__ == "__main__":
    app.run()

import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full")

# ─────────────────────────────────────────────────────────────────────────────
# LAB 13: THE TAIL LATENCY TRAP
#
# Chapter: Model Serving (@sec-model-serving)
# Core Invariant: Little's Law — L = λW.
#   Queue depth = arrival rate × sojourn time.
#   P99 latency is what users experience. Average latency is a misleading
#   metric for production SLOs. Tail latency is determined by queuing theory,
#   not just compute time.
#
# Two deployment contexts: Cloud (H100) vs Mobile (Smartphone NPU).
# 2-Act structure: ~35–40 minutes total.
#
# Act I  — The Queue Blindspot (12–15 min)
#   Stakeholder: Head of Platform — average looks fine, users complaining.
#   Prediction: what is wrong with relying on average latency at high load?
#   Instrument: Little's Law explorer (λ, μ sliders) showing P50/P95/P99/P99.9
#   Prediction-vs-reality overlay; Act I reflection (4-option radio).
#   MathPeek: M/M/1 equations, Little's Law derivation, P99 formula.
#
# Act II — The P99 Latency Histogram (FIRST INTRODUCTION) (20–25 min)
#   Stakeholder: SRE Team Lead — 10× traffic spike coming at launch.
#   Prediction: do we need more replicas before launch?
#   Instrument: P99 Latency Histogram — overlapping histograms (current vs 10×),
#   replica count, batch size, auto-scaling threshold.
#   FAILURE STATE: P99 > SLO target → danger callout.
#   Reflection: why is continuous batching essential for LLM serving?
#   MathPeek: latency percentile calculation, queuing under non-Poisson arrivals.
#
# New instrument first appearing here:
#   P99 Latency Histogram — dual overlapping bar charts with percentile markers.
#
# Design Ledger: chapter=13
# ─────────────────────────────────────────────────────────────────────────────


# ─── CELL 0: SETUP (hide_code=False — leave visible) ─────────────────────────
@app.cell
def _():
    import marimo as mo
    import sys
    from pathlib import Path
    import plotly.graph_objects as go
    import numpy as np
    import math

    _root = Path(__file__).resolve().parents[2]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

    from labs.core.state import DesignLedger
    from labs.core.style import COLORS, LAB_CSS, apply_plotly_theme

    ledger = DesignLedger()

    # ── Hardware constants ─────────────────────────────────────────────────────
    # Source for all constants: @sec-hardware-reference, NVIDIA H100 spec sheet,
    # Apple A17 Pro spec, model_serving.qmd hardware baselines.

    H100_BW_GBS      = 3350   # H100 SXM5 HBM3e bandwidth — NVIDIA official spec
    H100_TFLOPS_FP16 = 1979   # H100 FP16 tensor core TFLOPS — NVIDIA official spec
    H100_RAM_GB      = 80     # H100 HBM3e capacity — NVIDIA official spec
    H100_TDP_W       = 700    # H100 SXM5 TDP — NVIDIA official spec

    MOBILE_BW_GBS    = 68     # Apple A17 Pro class NPU memory bandwidth (GB/s)
    MOBILE_TOPS_INT8 = 35     # Apple A17 Pro NPU peak INT8 — Apple announced spec
    MOBILE_RAM_GB    = 8      # Smartphone NPU accessible DRAM (GB)
    MOBILE_TDP_W     = 5      # Smartphone sustained thermal budget (Watts)

    # ── Serving baseline constants (from @sec-model-serving-fundamentals) ─────
    # Act I scenario: 45 ms average service time, 100 ms P99 SLO
    ACT1_BASE_SERVICE_MS  = 45    # baseline avg service time at low utilization (ms)
    ACT1_SLO_P99_MS       = 100   # Act I P99 SLO target (ms)

    # Act II scenario: 50 ms avg service time, 200 ms P99 SLO, 1000 req/s base capacity
    ACT2_BASE_SERVICE_MS  = 50    # baseline avg service time per replica (ms)
    ACT2_SLO_P99_MS       = 200   # Act II P99 SLO target (ms)
    ACT2_BASE_CAPACITY    = 1000  # current serving capacity (req/s)
    ACT2_SPIKE_FACTOR     = 10    # launch-day expected traffic multiple

    return (
        mo, ledger, COLORS, LAB_CSS, apply_plotly_theme, go, np, math,
        H100_BW_GBS, H100_TFLOPS_FP16, H100_RAM_GB, H100_TDP_W,
        MOBILE_BW_GBS, MOBILE_TOPS_INT8, MOBILE_RAM_GB, MOBILE_TDP_W,
        ACT1_BASE_SERVICE_MS, ACT1_SLO_P99_MS,
        ACT2_BASE_SERVICE_MS, ACT2_SLO_P99_MS,
        ACT2_BASE_CAPACITY, ACT2_SPIKE_FACTOR,
    )


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(mo, LAB_CSS, COLORS):
    _cloud_c = COLORS["Cloud"]
    _mob_c   = COLORS["Mobile"]
    mo.vstack([
        LAB_CSS,
        mo.Html(f"""
        <div style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
                    padding: 36px 44px; border-radius: 16px; color: white;
                    box-shadow: 0 8px 32px rgba(0,0,0,0.3);">
            <div style="font-size: 0.72rem; font-weight: 700; letter-spacing: 0.18em;
                        color: #475569; text-transform: uppercase; margin-bottom: 10px;">
                Machine Learning Systems &middot; Volume I &middot; Lab 13
            </div>
            <h1 style="margin: 0 0 10px 0; font-size: 2.4rem; font-weight: 900;
                       color: #f8fafc; line-height: 1.1; letter-spacing: -0.02em;">
                The Tail Latency Trap
            </h1>
            <p style="margin: 0 0 20px 0; font-size: 1.05rem; color: #94a3b8;
                      max-width: 680px; line-height: 1.65;">
                Average latency looks healthy. P99 is silently on fire.
                Little&apos;s Law reveals that the queue &mdash; not the model &mdash;
                is the true source of user-visible slowness at high utilization.
            </p>
            <div style="display: flex; gap: 12px; flex-wrap: wrap;">
                <span style="background: rgba(99,102,241,0.15); color: #a5b4fc;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(99,102,241,0.25);">
                    Act I &middot; The Queue Blindspot &middot; 12&ndash;15 min
                </span>
                <span style="background: rgba(99,102,241,0.15); color: #a5b4fc;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(99,102,241,0.25);">
                    Act II &middot; Design Challenge &middot; 20&ndash;25 min
                </span>
                <span style="background: rgba(16,185,129,0.15); color: #6ee7b7;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(16,185,129,0.25);">
                    35&ndash;40 min total
                </span>
                <span style="background: rgba(245,158,11,0.15); color: #fcd34d;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(245,158,11,0.25);">
                    New instrument: P99 Latency Histogram
                </span>
            </div>
            <div style="display: flex; gap: 12px; flex-wrap: wrap; margin-top: 12px;">
                <span class="badge badge-info">L = &lambda;W</span>
                <span class="badge badge-warn">P99 &ne; average</span>
                <span class="badge badge-ok">SLO Act I: P99 &lt; 100 ms</span>
                <span class="badge badge-ok">SLO Act II: P99 &lt; 200 ms</span>
            </div>
        </div>
        """),
    ])
    return


# ─────────────────────────────────────────────────────────────────────────────
# RECOMMENDED READING
# ─────────────────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md("""
    **Recommended Reading** — Complete the following chapter sections before this lab:

    - @sec-model-serving-fundamentals — Request lifecycle, batching, queue theory basics
    - @sec-littles-law — Little's Law derivation: `L = λW` and the M/M/1 queue model
    - @sec-tail-latency — Why P99 diverges from average at high utilization
    - @sec-serving-slos — SLO definitions, P50/P95/P99/P99.9 percentile semantics
    - @sec-continuous-batching — How continuous batching differs from static batching
    """), kind="info")
    return


# ─────────────────────────────────────────────────────────────────────────────
# CONTEXT TOGGLE
# ─────────────────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(mo):
    context_toggle = mo.ui.radio(
        options={"Cloud (H100)": "cloud", "Mobile (Smartphone NPU)": "mobile"},
        value="Cloud (H100)",
        label="Deployment context:",
        inline=True,
    )
    mo.vstack([
        mo.md("---"),
        context_toggle,
    ])
    return (context_toggle,)


# ─────────────────────────────────────────────────────────────────────────────
# ACT I — THE QUEUE BLINDSPOT
# Scenario: monitoring dashboard says green. Users are complaining.
# ─────────────────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(mo, COLORS, context_toggle):
    _ctx    = context_toggle.value
    _color  = COLORS["Cloud"] if _ctx == "cloud" else COLORS["Mobile"]
    _bg     = COLORS["BlueL"] if _ctx == "cloud" else COLORS["OrangeL"]

    if _ctx == "cloud":
        _persona = "Head of Platform"
        _quote = (
            "Our average inference latency is 45 ms. We have an SLO of 100 ms P99. "
            "Our monitoring shows we are hitting the SLO 99% of the time. "
            "But users are complaining about slowness — complaint tickets are up 300% this week. "
            "The dashboard says green. Explain this."
        )
        _device_note = "Cloud inference cluster — H100 GPUs, stateless replicas, Poisson request arrivals."
    else:
        _persona = "Head of Mobile Platform"
        _quote = (
            "On-device NPU inference averages 45 ms. App store SLO is 100 ms P99. "
            "Our monitoring shows P99 is within SLO across 99% of sessions. "
            "App reviews say 'sometimes takes forever' and 1-star ratings are climbing. "
            "Engineering says the average looks fine. Who is wrong?"
        )
        _device_note = "Mobile NPU — single inference engine, no horizontal scaling, thermal constraints."

    mo.vstack([
        mo.md("---"),
        mo.md("## Act I — The Queue Blindspot"),
        mo.md(f"""
        Your serving system processes inference requests.
        The monitoring dashboard shows **average latency = 45 ms**, well below
        the P99 SLO of 100 ms. Customer complaints are spiking anyway.

        *{_device_note}*
        """),
        mo.Html(f"""
        <div style="border-left:4px solid {_color}; background:{_bg};
                    border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
            <div style="font-size:0.72rem; font-weight:700; color:{_color};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                Incoming Message &middot; {_persona}
            </div>
            <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                &ldquo;{_quote}&rdquo;
            </div>
        </div>
        """),
        mo.md("""
        The data is consistent. The SLO measurement is technically correct —
        P99 across all sessions is within 100 ms. Yet users experience much worse.
        Something is being hidden by the aggregation.
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
            "A) The monitoring is broken — average and P99 both look fine":
                "A",
            "B) The SLO is too loose — 100 ms P99 still means 1 in 100 requests is slow":
                "B",
            "C) At high load, P99 spikes much higher than average — Little's Law says "
            "queue depth grows superlinearly near capacity":
                "C",
            "D) The P99 measurement is correct — user complaints are unrelated to latency":
                "D",
        },
        label=(
            "Prediction Lock — Act I. "
            "Why does average latency = 45 ms and P99 SLO compliance = 99% "
            "coexist with widespread user complaints about slowness?"
        ),
    )
    mo.vstack([
        mo.Html("""
        <div style="background:#1e293b; border-radius:10px; padding:4px 18px 12px 18px;
                    border-left:4px solid #6366f1; margin-bottom:8px;">
            <div style="font-size:0.72rem; font-weight:700; color:#6366f1;
                        text-transform:uppercase; letter-spacing:0.1em;
                        margin-top:12px; margin-bottom:8px;">
                Prediction Lock &middot; Act I
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
# ACT I — LITTLE'S LAW EXPLORER
# Instruments: arrival rate slider, capacity slider, SLO target slider.
# Shows: queue depth L, W (mean wait), P50/P95/P99/P99.9, utilization rho.
# Warning when rho > 0.70.
# ─────────────────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(mo, act1_pred, ACT1_SLO_P99_MS):
    mo.stop(act1_pred.value is None)

    arrival_rate_slider = mo.ui.slider(
        start=50, stop=950, value=800, step=10,
        label="Arrival rate \u03bb (requests/second)",
        show_value=True,
    )
    capacity_slider = mo.ui.slider(
        start=100, stop=1000, value=1000, step=50,
        label="Processing capacity \u03bc (requests/second)",
        show_value=True,
    )
    slo_slider = mo.ui.slider(
        start=50, stop=500, value=ACT1_SLO_P99_MS, step=10,
        label="P99 SLO target (ms)",
        show_value=True,
    )

    mo.vstack([
        mo.md("### Simulator — Little's Law Explorer"),
        mo.md("""
        Drag the arrival rate toward the capacity limit and observe how the
        P99 latency diverges from the average. The SLO target line shows
        when your tail latency crosses the contractual threshold.

        *Key: watch what happens after utilization crosses 70%.*
        """),
        mo.hstack([arrival_rate_slider, capacity_slider], justify="start", gap="2rem"),
        slo_slider,
    ])
    return (arrival_rate_slider, capacity_slider, slo_slider)


@app.cell(hide_code=True)
def _(
    mo, act1_pred, arrival_rate_slider, capacity_slider, slo_slider,
    go, np, apply_plotly_theme, COLORS,
    ACT1_BASE_SERVICE_MS,
):
    mo.stop(act1_pred.value is None)

    # ── M/M/1 Queue Physics ───────────────────────────────────────────────────
    # Source: @sec-littles-law, Kleinrock (1975) "Queueing Systems Vol. 1"
    #
    # Little's Law (exact, no distributional assumption):
    #   L = lambda * W
    #   L — mean queue depth (requests in system)
    #   lambda — arrival rate (req/s)
    #   W — mean sojourn time (ms)
    #
    # M/M/1 mean sojourn time:
    #   W = 1 / (mu - lambda)   for lambda < mu  [rates in req/ms]
    #
    # M/M/1 percentile sojourn (exact CDF inversion):
    #   W_p = -ln(1 - p) / (mu - lambda)   [in ms, rates in req/ms]
    #   P99:   -ln(0.01) = 4.605
    #   P99.9: -ln(0.001) = 6.908

    _lambda_rps = float(arrival_rate_slider.value)
    _mu_rps     = float(capacity_slider.value)
    _slo_ms     = float(slo_slider.value)

    # Stable queue requires rho < 1
    _rho        = _lambda_rps / max(_mu_rps, 0.001)
    _rho        = min(_rho, 0.9999)

    # Convert to per-ms rates
    _lambda_ms  = _lambda_rps / 1000.0
    _mu_ms      = _mu_rps / 1000.0
    _gap_ms     = max(_mu_ms - _lambda_ms, 1e-9)

    # Mean sojourn: W = 1/(mu-lambda) ms
    _W_ms       = 1.0 / _gap_ms

    # Queue depth via Little's Law
    _L          = _lambda_ms * _W_ms

    # Percentile latencies
    _p50_ms     = -np.log(0.50) / _gap_ms
    _p95_ms     = -np.log(0.05) / _gap_ms
    _p99_ms     = -np.log(0.01) / _gap_ms
    _p999_ms    = -np.log(0.001) / _gap_ms

    # P99 at 80% utilization for annotation
    _mu_ms_80   = _mu_ms
    _gap_80     = _mu_ms_80 * (1.0 - 0.80)
    _p99_at_80  = 4.605 / max(_gap_80, 1e-9)

    # Color coding
    _rho_color = (COLORS["RedLine"] if _rho > 0.85
                  else COLORS["OrangeLine"] if _rho > 0.70
                  else COLORS["GreenLine"])
    _p99_color = (COLORS["RedLine"] if _p99_ms > _slo_ms
                  else COLORS["OrangeLine"] if _p99_ms > _slo_ms * 0.7
                  else COLORS["GreenLine"])

    # ── Metric cards ──────────────────────────────────────────────────────────
    _cards = f"""
    <div style="display: flex; gap: 16px; justify-content: center;
                flex-wrap: wrap; margin: 16px 0;">
        <div style="padding: 18px 22px; border: 1px solid #e2e8f0; border-radius: 10px;
                    min-width: 160px; text-align: center; background: white;
                    box-shadow: 0 2px 6px rgba(0,0,0,0.04);">
            <div style="color: #64748b; font-size: 0.78rem; font-weight: 600;
                        text-transform: uppercase; letter-spacing: 0.06em;">Utilization &rho;</div>
            <div style="font-size: 2rem; font-weight: 800;
                        color: {_rho_color}; margin-top: 4px;">{_rho * 100:.1f}%</div>
        </div>
        <div style="padding: 18px 22px; border: 1px solid #e2e8f0; border-radius: 10px;
                    min-width: 160px; text-align: center; background: white;
                    box-shadow: 0 2px 6px rgba(0,0,0,0.04);">
            <div style="color: #64748b; font-size: 0.78rem; font-weight: 600;
                        text-transform: uppercase; letter-spacing: 0.06em;">Avg wait (W)</div>
            <div style="font-size: 2rem; font-weight: 800;
                        color: {COLORS['GreenLine']}; margin-top: 4px;">{_W_ms:.1f} ms</div>
        </div>
        <div style="padding: 18px 22px; border: 1px solid #e2e8f0; border-radius: 10px;
                    min-width: 160px; text-align: center; background: white;
                    box-shadow: 0 2px 6px rgba(0,0,0,0.04);">
            <div style="color: #64748b; font-size: 0.78rem; font-weight: 600;
                        text-transform: uppercase; letter-spacing: 0.06em;">P99 latency</div>
            <div style="font-size: 2rem; font-weight: 800;
                        color: {_p99_color}; margin-top: 4px;">{_p99_ms:.0f} ms</div>
        </div>
        <div style="padding: 18px 22px; border: 1px solid #e2e8f0; border-radius: 10px;
                    min-width: 160px; text-align: center; background: white;
                    box-shadow: 0 2px 6px rgba(0,0,0,0.04);">
            <div style="color: #64748b; font-size: 0.78rem; font-weight: 600;
                        text-transform: uppercase; letter-spacing: 0.06em;">Queue depth (L)</div>
            <div style="font-size: 2rem; font-weight: 800;
                        color: {COLORS['BlueLine']}; margin-top: 4px;">{_L:.1f} req</div>
        </div>
        <div style="padding: 18px 22px; border: 1px solid #e2e8f0; border-radius: 10px;
                    min-width: 160px; text-align: center; background: white;
                    box-shadow: 0 2px 6px rgba(0,0,0,0.04);">
            <div style="color: #64748b; font-size: 0.78rem; font-weight: 600;
                        text-transform: uppercase; letter-spacing: 0.06em;">P99.9 latency</div>
            <div style="font-size: 2rem; font-weight: 800;
                        color: {COLORS['OrangeLine']}; margin-top: 4px;">{_p999_ms:.0f} ms</div>
        </div>
    </div>
    """

    # ── Latency percentile vs utilization chart ───────────────────────────────
    _rho_range = np.linspace(0.10, 0.990, 200)
    _gap_range = _mu_ms * (1.0 - _rho_range)
    _gap_range = np.maximum(_gap_range, 1e-12)

    _avg_c  = np.minimum(1.0 / _gap_range,           _slo_ms * 6)
    _p50_c  = np.minimum(-np.log(0.50) / _gap_range,  _slo_ms * 6)
    _p95_c  = np.minimum(-np.log(0.05) / _gap_range,  _slo_ms * 6)
    _p99_c  = np.minimum(-np.log(0.01) / _gap_range,  _slo_ms * 6)
    _p999_c = np.minimum(-np.log(0.001) / _gap_range, _slo_ms * 6)

    _y_max  = max(_slo_ms * 4.0, float(_p99_ms) * 1.5, 400.0)

    _fig1 = go.Figure()

    _fig1.add_trace(go.Scatter(
        x=_rho_range * 100, y=np.minimum(_avg_c, _y_max),
        mode="lines", name="Average (W)",
        line=dict(color=COLORS["BlueLine"], width=2),
    ))
    _fig1.add_trace(go.Scatter(
        x=_rho_range * 100, y=np.minimum(_p50_c, _y_max),
        mode="lines", name="P50",
        line=dict(color=COLORS["GreenLine"], width=2),
    ))
    _fig1.add_trace(go.Scatter(
        x=_rho_range * 100, y=np.minimum(_p95_c, _y_max),
        mode="lines", name="P95",
        line=dict(color=COLORS["OrangeLine"], width=2, dash="dot"),
    ))
    _fig1.add_trace(go.Scatter(
        x=_rho_range * 100, y=np.minimum(_p99_c, _y_max),
        mode="lines", name="P99",
        line=dict(color="#7c3aed", width=2.5),
    ))
    _fig1.add_trace(go.Scatter(
        x=_rho_range * 100, y=np.minimum(_p999_c, _y_max),
        mode="lines", name="P99.9",
        line=dict(color=COLORS["RedLine"], width=2),
    ))

    _fig1.add_hline(
        y=_slo_ms, line_dash="dash",
        line_color=COLORS["RedLine"], line_width=2,
        annotation_text=f"SLO = {_slo_ms:.0f} ms",
        annotation_position="bottom right",
        annotation_font_color=COLORS["RedLine"],
    )
    _fig1.add_vline(
        x=_rho * 100, line_dash="solid",
        line_color="#475569", line_width=1.5,
        annotation_text=f"\u03c1 = {_rho*100:.1f}%",
        annotation_position="top left",
        annotation_font_color="#475569",
    )
    _fig1.add_vrect(
        x0=70, x1=100,
        fillcolor="rgba(203,32,45,0.05)",
        layer="below", line_width=0,
    )

    _fig1.update_layout(
        title=dict(
            text=f"Latency Percentiles vs Utilization — \u03bb={_lambda_rps:.0f}, \u03bc={_mu_rps:.0f} req/s",
            font_size=14,
        ),
        xaxis_title="Server utilization \u03c1 (%)",
        yaxis_title="Latency (ms)",
        yaxis_range=[0, _y_max],
        showlegend=True,
        legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="left", x=0),
        height=360,
    )
    apply_plotly_theme(_fig1)

    # ── Physics block ─────────────────────────────────────────────────────────
    _formula = mo.md(f"""
    ```
    Little's Law:  L = \u03bb \u00d7 W
      \u03bb (arrival rate)      = {_lambda_rps:.0f} req/s = {_lambda_ms:.5f} req/ms
      \u03bc (service capacity)  = {_mu_rps:.0f} req/s = {_mu_ms:.5f} req/ms
      \u03c1 (utilization)       = \u03bb / \u03bc = {_lambda_rps:.0f} / {_mu_rps:.0f} = {_rho:.4f}

    M/M/1 mean sojourn:
      W = 1 / (\u03bc \u2212 \u03bb) = 1 / {_gap_ms:.6f} = {_W_ms:.1f} ms

    Queue depth:
      L = \u03bb \u00d7 W = {_lambda_ms:.5f} \u00d7 {_W_ms:.1f} = {_L:.2f} requests

    Percentile latencies  W_p = \u2212ln(1\u2212p) / (\u03bc\u2212\u03bb):
      P50  = {_p50_ms:.1f} ms
      P95  = {_p95_ms:.1f} ms
      P99  = {_p99_ms:.1f} ms       SLO = {_slo_ms:.0f} ms  \u2192  {"PASS" if _p99_ms <= _slo_ms else "FAIL"}
      P99.9 = {_p999_ms:.1f} ms

    At 80% utilization (scenario baseline for Act I):
      P99 = \u2212ln(0.01) / {_gap_80:.5f} = {_p99_at_80:.0f} ms
    ```
    """)

    _items = [mo.Html(_cards), _formula, mo.ui.plotly(_fig1)]

    if _rho > 0.70:
        _items.insert(0, mo.callout(mo.md(
            f"**Utilization {_rho*100:.1f}% exceeds 70% warning threshold.** "
            f"P99 = {_p99_ms:.0f} ms ({_p99_ms/max(_W_ms,0.001):.1f}\u00d7 average). "
            "Queue depth is growing nonlinearly. Every additional percent of "
            "utilization produces a larger absolute P99 increase."
        ), kind="warn"))

    mo.vstack(_items)
    return (_rho, _W_ms, _p99_ms, _p999_ms, _L, _lambda_rps, _mu_rps)


# ─────────────────────────────────────────────────────────────────────────────
# ACT I — MATHPEEK
# ─────────────────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(mo, act1_pred):
    mo.stop(act1_pred.value is None)

    mo.accordion({
        "The governing equations — Little's Law and M/M/1 Queue": mo.md("""
        **Little's Law** (John Little, 1961 — *Operations Research*):

        `L = \u03bb \u00d7 W`

        - **L** — mean number of requests in system (queue depth)
        - **\u03bb** — mean arrival rate (requests per unit time)
        - **W** — mean time a request spends in system (sojourn time)

        This is an **exact result** for any stable queuing system in steady state.
        No distributional assumption required.

        ---

        **M/M/1 Queue** — Poisson arrivals, exponential service, single server:

        `W = 1 / (\u03bc \u2212 \u03bb)`

        - \u03c1 = \u03bb / \u03bc (utilization fraction, must be < 1 for stability)
        - As \u03c1 \u2192 1, W \u2192 \u221e

        Little's Law substitution:
        `L = \u03bb / (\u03bc \u2212 \u03bb) = \u03c1 / (1 \u2212 \u03c1)`

        ---

        **Percentile latency — exact M/M/1 CDF inversion:**

        `W_p = \u2212ln(1 \u2212 p) / (\u03bc \u2212 \u03bb)`

        | Percentile | Factor |
        |-----------|--------|
        | P50       | 0.693  |
        | P95       | 3.000  |
        | P99       | 4.605  |
        | P99.9     | 6.908  |

        At \u03c1 = 0.80, \u03bc = 1000 req/s: gap = 200 req/s = 0.2 req/ms
        `W = 5 ms avg | P99 = 4.605/0.2 = 23 ms | P99.9 = 6.908/0.2 = 35 ms`

        At \u03c1 = 0.95, \u03bc = 1000 req/s: gap = 50 req/s = 0.05 req/ms
        `W = 20 ms avg | P99 = 4.605/0.05 = 92 ms | P99.9 = 6.908/0.05 = 138 ms`

        **Key insight:** The monitoring scenario (avg = 45 ms at \u03c1 = 80%) implies
        gap = 1/45 ms = 0.022 req/ms. P99 = 4.605/0.022 = **207 ms — 2\u00d7 the SLO**.
        The monitoring reports SLO compliance because it aggregates across
        low-utilization periods. At peak hours with \u03c1 = 80%, every session
        sees P99 = 207 ms.
        """),
    })
    return


# ─────────────────────────────────────────────────────────────────────────────
# ACT I — PREDICTION VS REALITY OVERLAY
# ─────────────────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(mo, act1_pred, ACT1_SLO_P99_MS):
    mo.stop(act1_pred.value is None)

    # Scenario reveal: avg = 45 ms implies rho = 80% with service_time = 9 ms
    # (W = svc / (1-rho) = 9 / 0.2 = 45 ms)
    # P99 = -ln(0.01) * W = 4.605 * 45 = 207 ms — > 100 ms SLO
    import math as _m1
    _W_scene    = 45.0    # stated average (ms)
    _p99_scene  = -_m1.log(0.01) * _W_scene   # 207 ms
    _ratio_scene = _p99_scene / _W_scene       # 4.605 (constant for M/M/1)

    _chosen  = act1_pred.value
    _correct = _chosen == "C"

    _expl = {
        "A": (
            "**The monitoring is not broken — it is measuring the wrong metric.** "
            "Average latency and per-session P99 can both show green while users "
            "at peak hours see tail spikes far above the SLO. "
            f"At 80% utilization with avg = {_W_scene:.0f} ms, "
            f"M/M/1 gives P99 = {_p99_scene:.0f} ms — "
            f"{_p99_scene / ACT1_SLO_P99_MS:.1f}\u00d7 the SLO. "
            "The monitoring reports accurately but aggregates across off-peak periods."
        ),
        "B": (
            "**The SLO threshold is not the problem — the measurement methodology is.** "
            "1 in 100 requests being slow is acceptable. The issue is that P99 at "
            f"peak utilization far exceeds 100 ms. With avg = {_W_scene:.0f} ms, "
            f"P99 = {_p99_scene:.0f} ms — not 100 ms. "
            "The SLO is not too loose; the system is overloaded at peak."
        ),
        "C": (
            f"**Correct.** "
            f"Average = {_W_scene:.0f} ms at 80% utilization. "
            f"M/M/1: P99 = \u2212ln(0.01) \u00d7 W = 4.605 \u00d7 {_W_scene:.0f} = "
            f"**{_p99_scene:.0f} ms** — {_p99_scene / ACT1_SLO_P99_MS:.1f}\u00d7 the SLO. "
            "The monitoring shows 'SLO compliance = 99%' because it samples across "
            "all utilization levels. At peak hours with consistent 80%+ utilization, "
            f"every user in that window sees P99 = {_p99_scene:.0f} ms."
        ),
        "D": (
            f"**User complaints are directly caused by tail latency.** "
            f"At 80% utilization, P99 = {_p99_scene:.0f} ms vs the 100 ms SLO. "
            "App reviews and support tickets correlate strongly with P99 spikes. "
            "The monitoring gives a misleading 'green' by aggregating across "
            "low-utilization periods where P99 is genuinely within SLO."
        ),
    }

    mo.callout(mo.md(
        f"**Prediction vs. Reality — Act I.** "
        f"You predicted: option {_chosen}. "
        f"\n\n{_expl[_chosen]}"
        f"\n\n**At 80% utilization with avg = {_W_scene:.0f} ms: P99 = {_p99_scene:.0f} ms** "
        f"({_p99_scene / ACT1_SLO_P99_MS:.1f}\u00d7 the SLO). "
        f"The M/M/1 P99/avg ratio is always \u2212ln(0.01) \u2248 {_ratio_scene:.2f}\u00d7."
    ), kind="success" if _correct else "warn")
    return


# ─────────────────────────────────────────────────────────────────────────────
# ACT I — REFLECTION
# ─────────────────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(mo, act1_pred):
    mo.stop(act1_pred.value is None)

    act1_reflect = mo.ui.radio(
        options={
            "A) Hardware degrades at high load — CPUs throttle when hot":
                "A",
            "B) Random request arrivals follow a Poisson distribution — rare bursts "
            "cause queuing that compounds near full utilization":
                "B",
            "C) P99 is just a statistical artifact with no physical meaning":
                "C",
            "D) High utilization causes CPU throttling, increasing service time":
                "D",
        },
        label=(
            "Reflection — Act I. "
            "Why does P99 latency spike near capacity even when average latency is fine?"
        ),
    )
    act1_reflect
    return (act1_reflect,)


@app.cell(hide_code=True)
def _(mo, act1_pred, act1_reflect):
    mo.stop(act1_pred.value is None)
    mo.stop(
        act1_reflect.value is None,
        mo.callout(mo.md("Select your answer to continue."), kind="warn"),
    )

    _r1  = act1_reflect.value
    _ok1 = _r1 == "B"

    _fb1 = {
        "A": (
            "**Hardware degradation is not the primary cause of M/M/1 tail spikes.** "
            "The P99 blow-up is a mathematical property of queuing theory, "
            "independent of hardware. Even with perfectly stable hardware, "
            "Poisson arrivals produce bursts that form queues near capacity. "
            "The gap (mu minus lambda) approaches zero, so P99 = 4.605/gap explodes."
        ),
        "B": (
            "**Correct.** Request arrivals are random — Poisson-distributed in many "
            "serving scenarios. On average, lambda < mu (stable). But by chance, "
            "k requests arrive in a short window, forming a transient queue. "
            "Near full utilization (rho near 1), the server drains queues slowly: "
            "drain rate = mu minus lambda approaches zero. "
            "P99 = 4.605/(mu minus lambda) shows this directly: as lambda approaches mu, "
            "the denominator vanishes and P99 diverges."
        ),
        "C": (
            "**P99 is a precise, measurable quantity.** "
            "It is the latency below which 99% of requests complete. "
            "For M/M/1: W_p = -ln(1-p)/(mu-lambda). "
            "The 1% of requests exceeding P99 are exactly the users filing complaints."
        ),
        "D": (
            "**CPU throttling is a hardware effect separate from queuing dynamics.** "
            "The M/M/1 tail spike occurs even on thermally stable hardware. "
            "It is purely mathematical: Poisson arrivals near capacity produce "
            "queue buildup because the drain rate (mu minus lambda) approaches zero. "
            "Throttling can compound the problem, but queuing theory alone "
            "explains the P99 divergence."
        ),
    }

    mo.callout(mo.md(f"**Reflection feedback.** {_fb1[_r1]}"),
               kind="success" if _ok1 else "warn")
    return


@app.cell(hide_code=True)
def _(mo, act1_pred):
    mo.stop(act1_pred.value is None)
    mo.callout(mo.md("""
    **Act I Takeaway — The Averaging Trap.**

    Little's Law (`L = \u03bbW`) is a lens, not a formula. Every time you see
    "average latency = X ms", ask: what is the utilization?

    The M/M/1 P99/average ratio is fixed at \u2212ln(0.01) \u2248 4.6\u00d7 regardless of
    utilization. But as utilization rises, the average rises — dragging P99
    well above any SLO. A monitoring system that only reports average latency
    is structurally incapable of detecting an SLO violation until it is catastrophic.

    **Rule:** SLOs must be defined on tail percentiles (P99, P99.9). Never on averages.
    """), kind="info")
    return


# ─────────────────────────────────────────────────────────────────────────────
# ACT II — THE P99 LATENCY HISTOGRAM (FIRST INTRODUCTION)
# Scenario: 10x launch-day traffic spike. Do we need more replicas?
# ─────────────────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(mo, COLORS, context_toggle):
    _ctx    = context_toggle.value
    _color  = COLORS["Cloud"] if _ctx == "cloud" else COLORS["Mobile"]
    _bg     = COLORS["BlueL"] if _ctx == "cloud" else COLORS["OrangeL"]

    if _ctx == "cloud":
        _persona = "SRE Team Lead"
        _quote = (
            "Launch day is in 48 hours. We expect a 10\u00d7 traffic spike. "
            "Current capacity: 1,000 requests/second. "
            "Current average latency: 50 ms. "
            "SLO: P99 < 200 ms. "
            "Engineering wants to wait and auto-scale reactively. "
            "Product wants to provision now. "
            "Do we need to provision more replicas before the spike, "
            "or will the system hold?"
        )
    else:
        _persona = "Mobile SRE Lead"
        _quote = (
            "Launch day is in 48 hours. We expect a 10\u00d7 spike in on-device "
            "inference requests. Current NPU capacity: 1,000 requests/second. "
            "Average inference time: 50 ms. SLO: P99 < 200 ms. "
            "We cannot add NPU capacity per device, but we can tune batch size "
            "and shed load via priority queues. Will the system hold at 10\u00d7?"
        )

    mo.vstack([
        mo.md("---"),
        mo.md("## Act II — The P99 Latency Histogram"),
        mo.md("""
        You understand why tail latency diverges from average near capacity.
        Your task: design a serving configuration that survives a 10\u00d7 traffic spike
        while meeting the P99 SLO.

        This act introduces the **P99 Latency Histogram** — a dual-histogram view
        showing the current latency distribution (blue, current load) alongside
        the 10\u00d7 spike distribution (red). Vertical lines mark P50, P95, P99, and P99.9.
        This is the standard capacity-planning instrument in production ML serving.
        """),
        mo.Html(f"""
        <div style="border-left:4px solid {_color}; background:{_bg};
                    border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
            <div style="font-size:0.72rem; font-weight:700; color:{_color};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                Incoming Message &middot; {_persona}
            </div>
            <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                &ldquo;{_quote}&rdquo;
            </div>
        </div>
        """),
    ])
    return


# ─────────────────────────────────────────────────────────────────────────────
# ACT II — PREDICTION LOCK
# ─────────────────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(mo):
    act2_pred = mo.ui.radio(
        options={
            "A) No — 50 ms average with 200 ms SLO gives plenty of headroom":
                "A",
            "B) Yes — at 10\u00d7 load without extra capacity, P99 will violate the SLO by more than 100\u00d7":
                "B",
            "C) Maybe — add 2\u00d7 capacity as a safety margin":
                "C",
            "D) No — modern load balancers prevent queue buildup":
                "D",
        },
        label=(
            "Prediction Lock — Act II. "
            "Current: 1,000 req/s capacity, avg = 50 ms, SLO = P99 < 200 ms. "
            "Launch day: 10,000 req/s expected. No extra capacity provisioned yet. "
            "Will the system meet the P99 SLO at 10\u00d7 load?"
        ),
    )
    mo.vstack([
        mo.Html("""
        <div style="background:#1e293b; border-radius:10px; padding:4px 18px 12px 18px;
                    border-left:4px solid #6366f1; margin-bottom:8px;">
            <div style="font-size:0.72rem; font-weight:700; color:#6366f1;
                        text-transform:uppercase; letter-spacing:0.1em;
                        margin-top:12px; margin-bottom:8px;">
                Prediction Lock &middot; Act II
            </div>
        </div>
        """),
        act2_pred,
    ])
    return (act2_pred,)


@app.cell(hide_code=True)
def _(mo, act2_pred):
    mo.stop(
        act2_pred.value is None,
        mo.callout(mo.md("Select your prediction to continue."), kind="warn"),
    )
    return


# ─────────────────────────────────────────────────────────────────────────────
# ACT II — INSTRUMENTS
# Sliders: replica count, batch size, auto-scaling threshold.
# ─────────────────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(mo, act2_pred, context_toggle, ACT2_BASE_CAPACITY, ACT2_SPIKE_FACTOR):
    mo.stop(act2_pred.value is None)

    _ctx        = context_toggle.value
    _spike_rps  = ACT2_BASE_CAPACITY * ACT2_SPIKE_FACTOR

    replicas_slider = mo.ui.slider(
        start=1, stop=20, value=1, step=1,
        label="Replica count",
        show_value=True,
    )
    batch_size_slider2 = mo.ui.slider(
        start=1, stop=32, value=1, step=1,
        label="Batch size (requests per forward pass)",
        show_value=True,
    )
    autoscale_slider = mo.ui.slider(
        start=50, stop=100, value=80, step=5,
        label="Auto-scaling trigger threshold (% utilization)",
        show_value=True,
    )

    if _ctx == "cloud":
        _note = (
            f"Each replica: {ACT2_BASE_CAPACITY:,} req/s capacity at baseline. "
            f"10\u00d7 spike = {_spike_rps:,} req/s."
        )
        _controls = mo.vstack([
            mo.md(f"**Configure your serving system for the 10\u00d7 spike:**"),
            mo.md(f"*{_note}*"),
            replicas_slider,
            batch_size_slider2,
            autoscale_slider,
        ])
    else:
        _note = (
            f"Mobile: parallel inference threads per device. "
            f"10\u00d7 spike = {_spike_rps:,} req/s aggregate."
        )
        _controls = mo.vstack([
            mo.md("**Configure your serving system for the 10\u00d7 spike:**"),
            mo.md(f"*{_note}*"),
            replicas_slider,
            batch_size_slider2,
            autoscale_slider,
            mo.callout(mo.md(
                "**Mobile constraint:** Beyond 4 threads, NPU thermal throttling "
                "degrades service time. More threads helps up to a point, then hurts."
            ), kind="warn"),
        ])

    _controls
    return (replicas_slider, batch_size_slider2, autoscale_slider)


# ─────────────────────────────────────────────────────────────────────────────
# ACT II — P99 LATENCY HISTOGRAM + LIVE TELEMETRY + FAILURE STATE
# NEW INSTRUMENT: Dual overlapping histogram (current vs 10x spike)
# ─────────────────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(
    mo, act2_pred, context_toggle,
    replicas_slider, batch_size_slider2, autoscale_slider,
    go, np, apply_plotly_theme, COLORS,
    ACT2_BASE_SERVICE_MS, ACT2_SLO_P99_MS,
    ACT2_BASE_CAPACITY, ACT2_SPIKE_FACTOR,
):
    mo.stop(act2_pred.value is None)

    import math as _m2

    _ctx            = context_toggle.value
    _replicas       = replicas_slider.value
    _batch          = batch_size_slider2.value
    _autoscale_pct  = autoscale_slider.value / 100.0

    # ── Per-replica physics ───────────────────────────────────────────────────
    # Source: @sec-batching-strategies
    # service_time(B) = service_time(1) * (1 + 0.12 * ln(B))
    # Coefficient 0.12 from HBM-bound transformer inference benchmarks.
    _svc_ms = ACT2_BASE_SERVICE_MS * (1.0 + 0.12 * _m2.log(max(_batch, 1)))

    # Mobile thermal throttle: beyond 4 threads, clock degrades
    _thermal_warn = False
    _throttle_msg = ""
    if _ctx == "mobile" and _replicas > 4:
        _thermal_warn = True
        _throttle      = 1.0 + 0.50 * min((_replicas - 4) / 16.0, 1.0)
        _svc_ms       *= _throttle
        _throttle_msg  = f"NPU thermal throttle: service time degraded to {_svc_ms:.0f} ms."

    # Per-replica capacity (req/s)
    _cap_per_r    = 1000.0 / _svc_ms
    _total_cap    = _cap_per_r * _replicas

    # Load levels
    _lambda_curr  = float(ACT2_BASE_CAPACITY)
    _lambda_spike = _lambda_curr * ACT2_SPIKE_FACTOR

    # Utilization
    _rho_curr     = min(_lambda_curr  / max(_total_cap, 0.001), 0.9999)
    _rho_spike    = min(_lambda_spike / max(_total_cap, 0.001), 0.9999)

    # Auto-scaling: reactive scale-out when utilization > threshold
    _autoscale_rep = 0
    if _rho_spike > _autoscale_pct:
        _needed_total   = _lambda_spike / _autoscale_pct
        _autoscale_rep  = max(0, int(_m2.ceil(_needed_total / _cap_per_r)) - _replicas)
    _eff_replicas   = _replicas + _autoscale_rep
    _eff_cap        = _cap_per_r * _eff_replicas
    _rho_spike_eff  = min(_lambda_spike / max(_eff_cap, 0.001), 0.9999)

    # ── M/M/1 percentile helper ───────────────────────────────────────────────
    def _pctile(p_val, lam_rps, cap_rps):
        """Exact M/M/1 percentile latency in ms."""
        _g = (cap_rps - lam_rps) / 1000.0   # gap in req/ms
        if _g <= 0:
            return 99999.0
        return -_m2.log(1.0 - p_val) / _g

    def _mean_w(lam_rps, cap_rps):
        g = (cap_rps - lam_rps) / 1000.0
        if g <= 0:
            return 99999.0
        return 1.0 / g

    _avg_curr   = _mean_w(_lambda_curr,  _total_cap)
    _avg_spike  = _mean_w(_lambda_spike, _eff_cap)

    _p50_curr   = _pctile(0.50,  _lambda_curr,  _total_cap)
    _p99_curr   = _pctile(0.99,  _lambda_curr,  _total_cap)
    _p999_curr  = _pctile(0.999, _lambda_curr,  _total_cap)

    _p50_spike  = _pctile(0.50,  _lambda_spike, _eff_cap)
    _p99_spike  = _pctile(0.99,  _lambda_spike, _eff_cap)
    _p999_spike = _pctile(0.999, _lambda_spike, _eff_cap)

    # Queue depths
    _L_curr  = (_lambda_curr  / 1000.0) * _avg_curr
    _L_spike = (_lambda_spike / 1000.0) * _avg_spike

    # Replicas needed to meet SLO without auto-scale
    # mu_total > lambda + 4.605*1000/SLO_ms
    _needed_cap_slo  = _lambda_spike + 4.605 * 1000.0 / ACT2_SLO_P99_MS
    _needed_replicas = int(_m2.ceil(_needed_cap_slo / _cap_per_r))

    # SLO compliance
    _slo_ok_curr  = _p99_curr  <= ACT2_SLO_P99_MS
    _slo_ok_spike = _p99_spike <= ACT2_SLO_P99_MS

    # Cost (cloud only: $3/hr per H100 — from @sec-cloud-cost)
    _cost_str = (f"${_eff_replicas * 3:.0f}/hr" if _ctx == "cloud" else "N/A (mobile)")

    # ── Color coding ──────────────────────────────────────────────────────────
    _c_p99_curr  = COLORS["GreenLine"] if _slo_ok_curr  else COLORS["RedLine"]
    _c_p99_spike = COLORS["GreenLine"] if _slo_ok_spike else COLORS["RedLine"]
    _c_rho_spike = (COLORS["RedLine"]    if _rho_spike_eff > 0.85
                    else COLORS["OrangeLine"] if _rho_spike_eff > 0.70
                    else COLORS["GreenLine"])

    # ── P99 Latency Histogram — FIRST INTRODUCTION ───────────────────────────
    # Dual overlapping bar charts: current load (blue) vs 10x spike (red).
    # Y-axis: probability mass per bin (density * bin_width).
    # Vertical lines: P50, P95, P99, P99.9 for each scenario.

    _rng    = np.random.default_rng(seed=77)
    _n      = 6000

    # Clip to avoid degenerate exponential samples
    _avg_c_clip = min(_avg_curr,  ACT2_SLO_P99_MS * 30)
    _avg_s_clip = min(_avg_spike, ACT2_SLO_P99_MS * 30)

    _samp_curr  = _rng.exponential(scale=_avg_c_clip, size=_n)
    _samp_spike = _rng.exponential(scale=_avg_s_clip, size=_n)

    _x_max = min(
        max(
            float(np.percentile(_samp_spike, 99.9)),
            ACT2_SLO_P99_MS * 5.0,
            300.0,
        ),
        ACT2_SLO_P99_MS * 25.0,
    )

    _bins    = np.linspace(0, _x_max, 52)
    _bw      = _bins[1] - _bins[0]
    _bin_mid = (_bins[:-1] + _bins[1:]) / 2.0

    _cnt_curr,  _ = np.histogram(_samp_curr,  bins=_bins, density=True)
    _cnt_spike, _ = np.histogram(_samp_spike, bins=_bins, density=True)

    _fig2 = go.Figure()

    # Current load bars (blue)
    _fig2.add_trace(go.Bar(
        x=_bin_mid,
        y=_cnt_curr * _bw,
        width=_bw * 0.82,
        marker_color=COLORS["BlueLine"],
        opacity=0.58,
        name=f"Current load ({_lambda_curr:,.0f} req/s)",
        hovertemplate="Latency: %{x:.0f} ms<br>Prob: %{y:.4f}<extra></extra>",
    ))

    # Spike load bars (red)
    _fig2.add_trace(go.Bar(
        x=_bin_mid,
        y=_cnt_spike * _bw,
        width=_bw * 0.82,
        marker_color=COLORS["RedLine"],
        opacity=0.55,
        name=f"10\u00d7 spike ({_lambda_spike:,.0f} req/s, {_eff_replicas} eff. replicas)",
        hovertemplate="Latency: %{x:.0f} ms<br>Prob: %{y:.4f}<extra></extra>",
    ))

    # SLO boundary
    _fig2.add_vline(
        x=ACT2_SLO_P99_MS,
        line_dash="dash",
        line_color=COLORS["RedLine"],
        line_width=2.5,
        annotation_text=f"SLO = {ACT2_SLO_P99_MS} ms",
        annotation_position="top right",
        annotation_font_color=COLORS["RedLine"],
        annotation_font_size=11,
    )

    # Percentile markers — current load
    for _pv, _pl, _pc in [
        (0.50, "P50 curr", COLORS["GreenLine"]),
        (0.99, "P99 curr", COLORS["BlueLine"]),
    ]:
        _xm = float(np.percentile(_samp_curr, _pv * 100))
        if _xm < _x_max:
            _fig2.add_vline(
                x=_xm, line_dash="dot",
                line_color=_pc, line_width=1.5,
                annotation_text=f"{_pl}={_xm:.0f}ms",
                annotation_position="top left",
                annotation_font_color=_pc,
                annotation_font_size=9,
            )

    # Percentile markers — spike load
    for _pv, _pl, _pc in [
        (0.99,  "P99 spike",  "#7c3aed"),
        (0.999, "P99.9 spike", COLORS["OrangeLine"]),
    ]:
        _xm2 = float(np.percentile(_samp_spike, _pv * 100))
        if _xm2 < _x_max:
            _fig2.add_vline(
                x=_xm2, line_dash="dot",
                line_color=_pc, line_width=1.5,
                annotation_text=f"{_pl}={_xm2:.0f}ms",
                annotation_position="bottom left",
                annotation_font_color=_pc,
                annotation_font_size=9,
            )

    _fig2.update_layout(
        title=dict(
            text=(f"P99 Latency Histogram — {_ctx.upper()} | "
                  f"replicas={_replicas}+{_autoscale_rep} auto | batch={_batch}"),
            font_size=13,
        ),
        xaxis_title="Request latency (ms)",
        yaxis_title="Probability mass per bin",
        barmode="overlay",
        showlegend=True,
        legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="left", x=0),
        height=400,
    )
    apply_plotly_theme(_fig2)

    # ── Metric cards ──────────────────────────────────────────────────────────
    _metric_html = f"""
    <div style="display: flex; gap: 14px; justify-content: center;
                flex-wrap: wrap; margin: 16px 0;">
        <div style="padding: 16px 20px; border: 1px solid #e2e8f0; border-radius: 10px;
                    min-width: 155px; text-align: center; background: white;">
            <div style="color: #64748b; font-size: 0.76rem; font-weight: 600;
                        text-transform: uppercase; letter-spacing: 0.05em;">P99 current</div>
            <div style="font-size: 1.85rem; font-weight: 800;
                        color: {_c_p99_curr}; margin-top: 4px;">
                {min(_p99_curr, 99999):.0f} ms</div>
        </div>
        <div style="padding: 16px 20px; border: 1px solid #e2e8f0; border-radius: 10px;
                    min-width: 155px; text-align: center; background: white;">
            <div style="color: #64748b; font-size: 0.76rem; font-weight: 600;
                        text-transform: uppercase; letter-spacing: 0.05em;">P99 at 10\u00d7 spike</div>
            <div style="font-size: 1.85rem; font-weight: 800;
                        color: {_c_p99_spike}; margin-top: 4px;">
                {min(_p99_spike, 99999):.0f} ms</div>
        </div>
        <div style="padding: 16px 20px; border: 1px solid #e2e8f0; border-radius: 10px;
                    min-width: 155px; text-align: center; background: white;">
            <div style="color: #64748b; font-size: 0.76rem; font-weight: 600;
                        text-transform: uppercase; letter-spacing: 0.05em;">Spike utilization</div>
            <div style="font-size: 1.85rem; font-weight: 800;
                        color: {_c_rho_spike}; margin-top: 4px;">
                {_rho_spike_eff * 100:.1f}%</div>
        </div>
        <div style="padding: 16px 20px; border: 1px solid #e2e8f0; border-radius: 10px;
                    min-width: 155px; text-align: center; background: white;">
            <div style="color: #64748b; font-size: 0.76rem; font-weight: 600;
                        text-transform: uppercase; letter-spacing: 0.05em;">Queue depth (spike)</div>
            <div style="font-size: 1.85rem; font-weight: 800;
                        color: {COLORS['BlueLine']}; margin-top: 4px;">
                {min(_L_spike, 99999):.1f} req</div>
        </div>
        <div style="padding: 16px 20px; border: 1px solid #e2e8f0; border-radius: 10px;
                    min-width: 155px; text-align: center; background: white;">
            <div style="color: #64748b; font-size: 0.76rem; font-weight: 600;
                        text-transform: uppercase; letter-spacing: 0.05em;">Replica cost</div>
            <div style="font-size: 1.85rem; font-weight: 800;
                        color: {COLORS['TextSec']}; margin-top: 4px;">
                {_cost_str}</div>
        </div>
    </div>
    """

    # ── Physics formula block ─────────────────────────────────────────────────
    _gap_rps = max(_eff_cap - _lambda_spike, 0.001)
    _formula = mo.md(f"""
    ```
    Serving physics at 10\u00d7 spike:

    \u03bb (spike)          = {ACT2_BASE_CAPACITY} \u00d7 {ACT2_SPIKE_FACTOR} = {_lambda_spike:,.0f} req/s
    service_time(B={_batch}) = {ACT2_BASE_SERVICE_MS} \u00d7 (1 + 0.12 \u00d7 ln({_batch})) = {_svc_ms:.1f} ms
    cap per replica    = 1000 / {_svc_ms:.1f} = {_cap_per_r:.1f} req/s
    eff. replicas      = {_replicas} (manual) + {_autoscale_rep} (auto) = {_eff_replicas}
    total capacity     = {_eff_replicas} \u00d7 {_cap_per_r:.1f} = {_eff_cap:.0f} req/s

    spike utilization  = {_lambda_spike:,.0f} / {_eff_cap:.0f} = {_rho_spike_eff:.4f}

    W_p99 = \u2212ln(1\u22120.99) / (\u03bc\u2212\u03bb)
      \u03bc\u2212\u03bb (req/ms) = ({_eff_cap:.0f}\u2212{_lambda_spike:,.0f}) / 1000 = {_gap_rps/1000:.4f}
      P99        = 4.605 / {_gap_rps/1000:.4f} = {min(_p99_spike, 99999):.0f} ms

    SLO: P99 < {ACT2_SLO_P99_MS} ms \u2192 {"PASS" if _slo_ok_spike else "FAIL"}
    Replicas needed to meet SLO (no auto-scale): {_needed_replicas}
    ```
    """)

    # ── Compose output with failure state ─────────────────────────────────────
    _widgets = [mo.Html(_metric_html), _formula, mo.ui.plotly(_fig2)]

    if not _slo_ok_spike:
        _fail = (
            f"**SLO VIOLATION at 10\u00d7 spike.** "
            f"P99 = {min(_p99_spike, 99999):.0f} ms | "
            f"SLO target = {ACT2_SLO_P99_MS} ms. "
            f"Add **{max(_needed_replicas - _eff_replicas, 1)} more replicas** "
            f"(total {_needed_replicas} required) "
            "or reduce batch size to lower service time."
        )
        _widgets.insert(0, mo.callout(mo.md(_fail), kind="danger"))

    if _thermal_warn:
        _idx = 1 if not _slo_ok_spike else 0
        _widgets.insert(_idx, mo.callout(mo.md(
            f"**Thermal throttle active (mobile).** "
            f"{_replicas} threads exceeds NPU sustained limit. "
            f"{_throttle_msg} "
            "Reduce thread count or implement duty-cycle scheduling."
        ), kind="warn"))

    if _slo_ok_spike and not _autoscale_rep:
        _widgets.insert(0, mo.callout(mo.md(
            f"**SLO met at 10\u00d7 spike.** "
            f"P99 = {_p99_spike:.0f} ms < {ACT2_SLO_P99_MS} ms. "
            f"Spike utilization: {_rho_spike_eff*100:.1f}%. "
            f"Cost: {_cost_str}."
        ), kind="success"))

    if _slo_ok_spike and _autoscale_rep > 0:
        _widgets.insert(0, mo.callout(mo.md(
            f"**Auto-scaling triggered** — utilization exceeded {autoscale_slider.value}%. "
            f"Added {_autoscale_rep} replicas (effective total: {_eff_replicas}). "
            f"P99 = {_p99_spike:.0f} ms — SLO met. "
            f"Cost: {_cost_str}. "
            "Note: reactive auto-scaling has 30–120 s scale-out lag; the queue "
            "builds during that window before new replicas are ready."
        ), kind="info"))

    mo.vstack(_widgets)
    return (
        _slo_ok_spike, _rho_spike_eff, _p99_spike, _avg_spike,
        _L_spike, _needed_replicas, _eff_replicas,
        _p99_curr, _rho_curr,
    )


# ─────────────────────────────────────────────────────────────────────────────
# ACT II — PREDICTION VS REALITY OVERLAY
# ─────────────────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(mo, act2_pred, ACT2_SLO_P99_MS, ACT2_BASE_CAPACITY, ACT2_SPIKE_FACTOR):
    mo.stop(act2_pred.value is None)

    import math as _m3

    # At 10x load with 1 replica (no provisioning):
    # lambda = 10,000 req/s; capacity = 1,000 req/s -> rho >> 1 -> queue unstable
    # Near saturation at rho = 0.9999:
    # gap = capacity * (1 - 0.9999) = 1000 * 0.0001 = 0.1 req/s = 0.0001 req/ms
    # P99 = 4.605 / 0.0001 = 46,050 ms — 230x the 200ms SLO
    _lambda_sp = float(ACT2_BASE_CAPACITY * ACT2_SPIKE_FACTOR)
    _cap_1r    = float(ACT2_BASE_CAPACITY)
    _gap_1r    = _cap_1r * (1.0 - 0.9999) / 1000.0   # req/ms
    _p99_1r    = 4.605 / max(_gap_1r, 1e-9)

    _chosen2  = act2_pred.value
    _correct2 = _chosen2 == "B"

    _exp2 = {
        "A": (
            "**Dangerously wrong — 50 ms average does not transfer to 10\u00d7 load.** "
            f"At 10\u00d7 traffic with 1 replica, \u03c1 = {_lambda_sp:,.0f}/{_cap_1r:,.0f} >> 1. "
            f"The queue is unstable. Near saturation, "
            f"P99 \u2248 **{_p99_1r:,.0f} ms** — {_p99_1r / ACT2_SLO_P99_MS:.0f}\u00d7 the SLO. "
            "The 'headroom' disappears because W = 1/(\u03bc\u2212\u03bb) diverges as \u03bb approaches \u03bc. "
            "50 ms average at current load means \u03bc\u2212\u03bb = 20 req/ms. "
            "At 10\u00d7 load with no new capacity, \u03bc\u2212\u03bb approaches zero."
        ),
        "B": (
            f"**Correct.** "
            f"At 10\u00d7 load with a single replica (capacity = {_cap_1r:,.0f} req/s), "
            f"\u03c1 = {_lambda_sp:,.0f}/{_cap_1r:,.0f} >> 1 — the queue is unstable. "
            f"Near saturation at \u03c1 = 99.99%: "
            f"P99 \u2248 4.605 / (\u03bc\u00d7(1\u2212\u03c1)) \u2248 **{_p99_1r:,.0f} ms** — "
            f"{_p99_1r / ACT2_SLO_P99_MS:.0f}\u00d7 the SLO. "
            "You must provision before launch. Reactive auto-scaling has "
            "30–120 s lag: during that window the queue floods and P99 catastrophically "
            "violates the SLO before new replicas are available."
        ),
        "C": (
            "**Underestimates the requirement.** "
            f"2\u00d7 capacity (2 replicas) = {_cap_1r * 2:,.0f} req/s. "
            f"At 10\u00d7 load = {_lambda_sp:,.0f} req/s, \u03c1 = {_lambda_sp / (_cap_1r * 2):.2f} — still >> 1. "
            "The system still saturates. You need roughly "
            f"{int(_m3.ceil(_lambda_sp / _cap_1r * 1.2))} replicas to "
            "maintain \u03c1 < 0.83 and keep P99 within SLO. "
            "The correct answer is 'yes, provision more' — but 2\u00d7 is not nearly enough."
        ),
        "D": (
            "**Load balancers distribute traffic; they do not create capacity.** "
            f"With 1 replica, all {_lambda_sp:,.0f} req/s hit the same server. "
            "The load balancer cannot reduce total arrival rate or add service capacity. "
            "Queue buildup depends on arrival rate vs. service rate — "
            "the load balancer changes neither."
        ),
    }

    mo.callout(mo.md(
        f"**Prediction vs. Reality — Act II.** You predicted: option {_chosen2}. "
        f"\n\n{_exp2[_chosen2]}"
    ), kind="success" if _correct2 else "warn")
    return


# ─────────────────────────────────────────────────────────────────────────────
# ACT II — MATHPEEK
# ─────────────────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(mo, act2_pred):
    mo.stop(act2_pred.value is None)

    mo.accordion({
        "The governing equations — Capacity planning and tail latency": mo.md("""
        **Required capacity for P99 SLO:**

        From W_p99 < SLO, invert to find required gap:

        `W_p99 = \u2212ln(0.01) / (\u03bc \u2212 \u03bb) < SLO_ms`

        Rearranging (rates in req/ms):
        `\u03bc \u2212 \u03bb > 4.605 / SLO_ms`

        Required total capacity (req/s):
        `\u03bc_total > \u03bb + 4.605 \u00d7 1000 / SLO_ms`

        For SLO = 200 ms, \u03bb = 10,000 req/s:
        `\u03bc_total > 10,000 + 4.605 \u00d7 1000 / 200 = 10,023 req/s`

        With service_time = 50 ms per replica (\u03bc = 20 req/s per replica):
        `replicas \u2265 ceil(10,023 / 20) = 502 replicas`

        ---

        **Non-Poisson arrivals (real burst traffic):**

        M/G/1 Pollaczek-Khinchine formula:
        `E[W] = \u03c1\u00b2 / (2(1\u2212\u03c1)) \u00d7 (1 + C_s\u00b2) + 1/\u03bc`

        C_s = coefficient of variation of service time.
        For M/M/1: C_s = 1. For deterministic service (M/D/1): C_s = 0.
        Bursty arrivals (C_a > 1 in M[X]/M/1) further inflate tail beyond M/M/1.

        Real launch-day traffic is often burstier than Poisson:
        C_a > 1. The M/M/1 prediction is a lower bound on actual P99.

        ---

        **Continuous batching** (from @sec-continuous-batching):

        Static batching: all sequences finish before any result returns.
        Short requests blocked behind long ones (head-of-line blocking).

        Continuous batching (vLLM, TGI — iteration-level scheduling):
        New requests join the decode loop at each token step.
        Short sequences complete without waiting for long ones.
        GPU utilization stays near static-batch levels.
        P99 for short requests approaches single-request latency.

        Kwon et al. (2023): 10–23\u00d7 throughput improvement over naive static batching.
        """),
    })
    return


# ─────────────────────────────────────────────────────────────────────────────
# ACT II — CONTINUOUS BATCHING REFLECTION
# ─────────────────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(mo, act2_pred):
    mo.stop(act2_pred.value is None)

    act2_reflect = mo.ui.radio(
        options={
            "A) It increases GPU memory efficiency by packing more tokens per batch":
                "A",
            "B) It decouples prefill (compute-bound) from decode (memory-bound) phases, "
            "allowing mixed-urgency batching":
                "B",
            "C) It always reduces average latency by processing more requests per second":
                "C",
            "D) It enables larger models to fit in memory through memory sharing":
                "D",
        },
        label=(
            "Reflection — Act II. "
            "Why is continuous batching essential for LLM serving at scale?"
        ),
    )
    act2_reflect
    return (act2_reflect,)


@app.cell(hide_code=True)
def _(mo, act2_pred, act2_reflect):
    mo.stop(act2_pred.value is None)
    mo.stop(
        act2_reflect.value is None,
        mo.callout(mo.md("Select your answer to continue."), kind="warn"),
    )

    _r2  = act2_reflect.value
    _ok2 = _r2 == "B"

    _fb2 = {
        "A": (
            "**Memory efficiency is a secondary benefit.** "
            "Paged attention (vLLM) improves KV-cache utilization, "
            "but that is orthogonal to continuous batching. "
            "The core P99 benefit of continuous batching is eliminating head-of-line "
            "blocking: short sequences no longer wait behind long-running ones."
        ),
        "B": (
            "**Correct.** "
            "LLM inference has two distinct phases: "
            "the *prefill* phase (input prompt processing — compute-bound, "
            "high arithmetic intensity) and the *decode* phase (autoregressive "
            "token generation — memory-bandwidth bound, low arithmetic intensity). "
            "Continuous batching uses iteration-level scheduling to mix requests at "
            "different decode steps in the same batch. A short new request joins the "
            "next decode iteration immediately — it does not wait for a 2,000-token "
            "generation to complete. This decoupling protects P99 for short requests "
            "while maintaining high GPU utilization for long-running ones."
        ),
        "C": (
            "**Not always.** "
            "Continuous batching can increase average latency for long sequences "
            "because short requests preempt decode steps. "
            "The benefit is to P99 for short requests, not necessarily average throughput. "
            "Operators tune continuous batching parameters to balance P99 for "
            "interactive workloads against throughput for batch workloads."
        ),
        "D": (
            "**Memory sharing is a separate optimization.** "
            "KV-cache sharing (prefix caching, RadixAttention) allows requests with "
            "shared prompts to reuse cached activations. That is distinct from "
            "iteration-level scheduling. Continuous batching does not change "
            "model size — it changes when and how requests are grouped for forward passes."
        ),
    }

    mo.callout(mo.md(f"**Reflection feedback.** {_fb2[_r2]}"),
               kind="success" if _ok2 else "warn")
    return


# ─────────────────────────────────────────────────────────────────────────────
# DESIGN LEDGER SAVE
# ─────────────────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(
    mo, ledger,
    act1_pred, act2_pred,
    context_toggle,
    arrival_rate_slider, capacity_slider,
    replicas_slider, batch_size_slider2,
    ACT2_BASE_SERVICE_MS, ACT2_SLO_P99_MS,
    ACT2_BASE_CAPACITY, ACT2_SPIKE_FACTOR,
):
    mo.stop(act1_pred.value is None or act2_pred.value is None)

    import math as _ml

    _ctx_val    = context_toggle.value
    _arr_rate   = float(arrival_rate_slider.value)
    _cap_rate   = float(capacity_slider.value)

    # Recompute final Act II P99 for ledger
    _rep_l     = replicas_slider.value
    _bat_l     = batch_size_slider2.value
    _svc_l     = ACT2_BASE_SERVICE_MS * (1.0 + 0.12 * _ml.log(max(_bat_l, 1)))
    _cap_r_l   = 1000.0 / _svc_l
    _total_l   = _cap_r_l * _rep_l
    _lam_sp    = float(ACT2_BASE_CAPACITY * ACT2_SPIKE_FACTOR)
    _rho_l     = min(_lam_sp / max(_total_l, 0.001), 0.9999)
    _gap_l     = max((_total_l - _lam_sp) / 1000.0, 1e-9)
    _p99_l     = 4.605 / _gap_l
    _slo_met   = _p99_l <= ACT2_SLO_P99_MS

    # Act I utilization
    _rho_a1    = min(_arr_rate / max(_cap_rate, 0.001), 0.9999)

    ledger.save(chapter=13, design={
        "context":          _ctx_val,
        "arrival_rate":     float(_arr_rate),
        "capacity":         float(_cap_rate),
        "utilization":      float(_rho_a1),
        "p99_latency_ms":   float(min(_p99_l, 99999.0)),
        "act1_prediction":  str(act1_pred.value),
        "act1_correct":     bool(act1_pred.value == "C"),
        "act2_result":      float(min(_p99_l, 99999.0)),
        "act2_decision":    f"replicas={_rep_l},batch={_bat_l}",
        "constraint_hit":   bool(not _slo_met),
        "slo_met":          bool(_slo_met),
    })

    mo.callout(mo.md(
        f"**Design Ledger updated — Chapter 13 saved.** "
        f"Context: `{_ctx_val}`. "
        f"Act I prediction: `{act1_pred.value}` "
        f"({'correct' if act1_pred.value == 'C' else 'incorrect'}). "
        f"Act II P99 at 10\u00d7 spike: {min(_p99_l, 99999):.0f} ms. "
        f"SLO {'met' if _slo_met else 'violated'}. "
        "Proceed to **Lab 14: ML Operations**."
    ), kind="success")
    return


# ─────────────────────────────────────────────────────────────────────────────
# KEY TAKEAWAYS
# ─────────────────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(mo):
    mo.vstack([
        mo.md("---"),
        mo.md("## Key Takeaways"),
        mo.callout(mo.md("""
        **1. Average latency is a lie.** Never define an SLO on average latency.
        At 80% utilization, M/M/1 gives P99 = 4.6\u00d7 average — placing the tail
        well above any reasonable SLO even when the average looks healthy.
        The correct metric is a tail percentile. Monitor what users experience,
        not what the infrastructure produces on average.
        """), kind="info"),
        mo.callout(mo.md("""
        **2. Little's Law is a capacity planning tool.** `L = \u03bbW` gives three levers:
        reduce arrival rate (\u03bb), reduce service time (quantization, faster model),
        or add replicas to reduce queue depth. Before any traffic event, compute
        the required capacity: `\u03bc > \u03bb + 4.605 / SLO_ms` (in req/ms units).
        Provision proactively. Reactive auto-scaling has 30–120 s lag — the queue
        floods before new replicas are ready. The constraint determines the lever.
        """), kind="info"),
    ])
    return


# ─────────────────────────────────────────────────────────────────────────────
# CONNECTIONS
# ─────────────────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(mo):
    mo.vstack([
        mo.md("---"),
        mo.md("""
        ## Connections

        **Textbook:** This lab explores the queuing theory foundations in @sec-model-serving
        and the tail latency analysis in @sec-tail-latency. The continuous batching
        reflection connects to @sec-continuous-batching. Little's Law is derived
        formally in @sec-littles-law.

        **TinyTorch:** In Module 13, you implement a request scheduler with configurable
        batching strategy. See `tinytorch/src/13_serving/`. The module includes a
        discrete-event simulator for validating your scheduler's P99 behavior under
        load, allowing you to verify the M/M/1 predictions from this lab empirically.

        **Next Lab:** Lab 14 (ML Operations) builds on this by exploring how monitoring
        systems detect the drift that changes the arrival rate distribution, converting
        a healthy serving system into a P99 violator over time without any code changes.
        """),
    ])
    return


# ─────────────────────────────────────────────────────────────────────────────
# DESIGN LEDGER HUD FOOTER
# ─────────────────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(mo, ledger, COLORS):
    _track     = ledger.get_track() or "NONE"
    _color_map = {
        "cloud":  COLORS["Cloud"],
        "edge":   COLORS["RedLine"],
        "mobile": COLORS["Mobile"],
        "tiny":   COLORS["Tiny"],
        "NONE":   "#475569",
    }
    _hud_color  = _color_map.get(_track, "#475569")
    _hud_status = "Uninitialized" if _track == "NONE" else "Active \u2014 Chapter 13"

    mo.Html(f"""
    <div style="display:flex; gap:28px; align-items:center; padding:12px 24px;
                background:#0f172a; border-radius:10px; margin-top:32px;
                font-family:'SF Mono','Fira Code',monospace; font-size:0.8rem;
                border:1px solid #1e293b;">
        <div style="color:#475569; font-weight:600; letter-spacing:0.06em;">DESIGN LEDGER</div>
        <div>
            <span style="color:#475569;">Context: </span>
            <span style="color:{_hud_color}; font-weight:700;">{_track.upper()}</span>
        </div>
        <div>
            <span style="color:#475569;">Chapter: </span>
            <span style="color:#e2e8f0;">13</span>
        </div>
        <div>
            <span style="color:#475569;">Invariant: </span>
            <span style="color:#e2e8f0;">L = \u03bbW</span>
        </div>
        <div>
            <span style="color:#475569;">Instrument: </span>
            <span style="color:#e2e8f0;">P99 Latency Histogram</span>
        </div>
        <div>
            <span style="color:#475569;">Status: </span>
            <span style="color:{'#4ade80' if _track != 'NONE' else '#f87171'};">{_hud_status}</span>
        </div>
    </div>
    """)
    return


if __name__ == "__main__":
    app.run()

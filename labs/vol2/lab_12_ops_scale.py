import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full")

# ─────────────────────────────────────────────────────────────────────────────
# LAB V2-12: THE SLO COMPOSITION TRAP
#
# Volume II, Chapter 12 — ML Operations at Scale
#
# Core Invariant: SLO budget allocation and cascading failure
#   If your end-to-end P99 SLO is 500ms across 5 services, you CANNOT allocate
#   100ms to each service. Tail latency composition means independent per-service
#   P99 compliance does NOT guarantee end-to-end SLO compliance.
#
# 2 Contexts:
#   Kubernetes  — Managed container orchestration (30s HPA response time)
#   Bare Metal  — Direct server management (manual scaling, faster circuit breakers)
#
# Act I  (12–15 min): SLO Composition Calculator
#   Stakeholder: SRE Lead with meeting-all-per-service-SLOs-but-violating-e2e
#   Instruments: number of services, per-service violation rate, correlation
#   Prediction: why end-to-end SLO is violated even when all per-service SLOs are met
#   Overlay: student prediction vs actual compound violation probability
#   Reflection: correct error budget allocation
#
# Act II (20–25 min): Cascading Failure Simulator
#   Stakeholder: Platform Reliability Lead post-6-hour-outage
#   Instruments: initial overload %, timeout, retry multiplier, circuit breaker threshold
#   Toggles: circuit breakers on/off, bulkheads on/off
#   Failure state: all services overloaded = CASCADE FAILURE danger callout
#   Reflection: why load shedding prevents cascades better than queuing
#
# Design Ledger: saves chapter="v2_12"
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

    # ── Hardware and operational constants ─────────────────────────────────────
    H100_BW_GBS         = 3350   # GB/s HBM3e — NVIDIA H100 SXM5 spec
    H100_TFLOPS_FP16    = 1979   # TFLOPS tensor core FP16 — NVIDIA spec
    K8S_SCALE_LAG_SEC   = 30     # Kubernetes HPA typical response time — k8s docs
    K8S_POD_START_SEC   = 5      # Pod startup time (warm image) — k8s production benchmark
    CIRCUIT_BREAKER_MS  = 100    # Circuit breaker response time — typical SRE practice

    ledger = DesignLedger()
    return (
        mo, ledger, COLORS, LAB_CSS, apply_plotly_theme,
        go, np, math,
        H100_BW_GBS, H100_TFLOPS_FP16,
        K8S_SCALE_LAG_SEC, K8S_POD_START_SEC, CIRCUIT_BREAKER_MS,
    )


# ─── CELL 1: HEADER (hide_code=True) ──────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, LAB_CSS, COLORS):
    _k8s_color = COLORS["Cloud"]
    mo.vstack([
        LAB_CSS,
        mo.Html(f"""
        <div style="background: linear-gradient(135deg, #0f172a 0%, #1a1a2e 55%, #16213e 100%);
                    padding: 36px 44px; border-radius: 16px; color: white;
                    box-shadow: 0 8px 32px rgba(0,0,0,0.4);">
            <div style="font-size: 0.72rem; font-weight: 700; letter-spacing: 0.18em;
                        color: #475569; text-transform: uppercase; margin-bottom: 10px;">
                Machine Learning Systems · Volume II · Lab 12
            </div>
            <h1 style="margin: 0 0 10px 0; font-size: 2.4rem; font-weight: 900;
                       color: #f8fafc; line-height: 1.1; letter-spacing: -0.02em;">
                The SLO Composition Trap
            </h1>
            <p style="margin: 0 0 22px 0; font-size: 1.05rem; color: #94a3b8;
                      max-width: 660px; line-height: 1.65;">
                Every per-service SLO is green. The end-to-end SLO is red. How?
                Tail latency compounds probabilistically across service chains — and
                when overload hits, queued requests amplify into cascading failures
                that bring down entire platforms in under 30 seconds.
            </p>
            <div style="display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 18px;">
                <span style="background: rgba(99,102,241,0.18); color: #a5b4fc;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(99,102,241,0.3);">
                    Act I: SLO Composition · Act II: Cascade Failure
                </span>
                <span style="background: rgba(16,185,129,0.15); color: #6ee7b7;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(16,185,129,0.25);">
                    35–40 min
                </span>
                <span style="background: rgba(245,158,11,0.15); color: #fcd34d;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(245,158,11,0.25);">
                    Requires: @sec-ops-scale
                </span>
            </div>
            <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                <span class="badge badge-info">Context A: Kubernetes (HPA 30s lag)</span>
                <span class="badge badge-info">Context B: Bare Metal (manual scale)</span>
                <span class="badge badge-warn">Invariant: P(e2e violation) &gt; per-service rate</span>
                <span class="badge badge-fail">Invariant: Queued load amplifies cascades</span>
            </div>
        </div>
        """),
    ])
    return


# ─── CELL 2: RECOMMENDED READING (hide_code=True) ─────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md("""
    **Recommended Reading** — Complete the following before this lab:

    - **@sec-ml-operations-scale-singlemodel-platform-operations-db8e** — The N-Models Problem:
      why operational complexity grows super-linearly with model count, and how shared platforms
      amortize per-model operational cost.
    - **@sec-ops-scale** (SLO Composition section) — How tail latency composition formula
      `P(total > SLO) = 1 - Π(1 - P_i)` governs end-to-end reliability budgets.
    - **@sec-ops-scale** (Cascading Failures section) — Circuit breaker state machines,
      bulkhead isolation patterns, and retry storm amplification mechanics.

    If you have not read these sections, the predictions in this lab will not map to the physics.
    """), kind="info")
    return


# ─── CELL 3: CONTEXT TOGGLE (hide_code=True) ──────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    context_toggle = mo.ui.radio(
        options={
            "Kubernetes (HPA auto-scaling)": "kubernetes",
            "Bare Metal (manual scaling)": "bare_metal",
        },
        value="Kubernetes (HPA auto-scaling)",
        label="Deployment context:",
        inline=True,
    )
    mo.vstack([
        mo.md("---"),
        mo.md("### Select your deployment context to orient the instruments:"),
        context_toggle,
    ])
    return (context_toggle,)


@app.cell(hide_code=True)
def _(mo, context_toggle, COLORS, K8S_SCALE_LAG_SEC, K8S_POD_START_SEC):
    _ctx = context_toggle.value
    _is_k8s = _ctx == "kubernetes"
    _color = COLORS["Cloud"] if _is_k8s else COLORS["GreenLine"]
    _label = "Kubernetes (HPA auto-scaling)" if _is_k8s else "Bare Metal (manual scaling)"
    _specs = (
        f"HPA response time: {K8S_SCALE_LAG_SEC}s · Pod startup: {K8S_POD_START_SEC}s · "
        "Circuit breakers: sidecar proxy (Envoy/Istio) · Bulkheads: namespace isolation"
        if _is_k8s else
        "Scale response: manual (minutes) · Process start: <1s · "
        "Circuit breakers: application-level · Bulkheads: process isolation"
    )
    mo.Html(f"""
    <div style="border-left: 4px solid {_color}; background: {'#f0f4ff' if _is_k8s else '#ecfdf5'};
                border-radius: 0 10px 10px 0; padding: 14px 20px; margin: 10px 0;">
        <div style="font-size: 0.72rem; font-weight: 700; color: {_color};
                    text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 4px;">
            Active Context
        </div>
        <div style="font-weight: 700; font-size: 1.05rem; color: #1e293b;">{_label}</div>
        <div style="font-size: 0.85rem; color: #475569; margin-top: 3px;">{_specs}</div>
    </div>
    """)
    return


# ═════════════════════════════════════════════════════════════════════════════
# ACT I: THE SLO COMPOSITION MISTAKE
# Stakeholder: SRE Lead | Prediction: why e2e SLO is violated
# ═════════════════════════════════════════════════════════════════════════════

@app.cell(hide_code=True)
def _(mo):
    mo.Html("""
    <div style="margin: 28px 0 8px 0;">
        <div style="font-size: 0.72rem; font-weight: 700; letter-spacing: 0.14em;
                    text-transform: uppercase; color: #94a3b8; margin-bottom: 6px;
                    display: flex; align-items: center; gap: 8px;">
            <span style="background: #006395; color: white; border-radius: 50%;
                         width: 22px; height: 22px; display: inline-flex; align-items: center;
                         justify-content: center; font-size: 0.72rem; font-weight: 800;">I</span>
            Act I — Calibration · 12–15 min
            <span style="flex: 1; height: 1px; background: #e2e8f0;"></span>
        </div>
        <div style="font-size: 1.5rem; font-weight: 800; color: #0f172a; line-height: 1.2;">
            The SLO Composition Mistake
        </div>
        <div style="font-size: 0.95rem; color: #475569; margin-top: 4px;">
            Meeting all per-service SLOs is necessary but not sufficient for end-to-end SLO compliance.
        </div>
    </div>
    """)
    return


# ─── ACT I: STAKEHOLDER MESSAGE ───────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, COLORS):
    _color = COLORS["BlueLine"]
    mo.Html(f"""
    <div style="border-left: 4px solid {_color}; background: {COLORS['BlueL']};
                border-radius: 0 10px 10px 0; padding: 16px 22px; margin: 12px 0;">
        <div style="font-size: 0.72rem; font-weight: 700; color: {_color};
                    text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px;">
            Incoming Message · SRE Lead, Platform Reliability
        </div>
        <div style="font-style: italic; font-size: 1.0rem; color: #1e293b; line-height: 1.65;">
            "Our end-to-end API SLO is P99 &lt; 500ms. We have 5 microservices in the critical
            path: API gateway, auth service, feature store, model inference, and response
            formatter. We allocated P99 = 100ms to each service — that adds up to 500ms
            perfectly. All five dashboards show green. But the end-to-end P99 is 547ms
            and we are violating our customer SLA 3% of the time. Our engineers say every
            service is meeting its SLO. What is wrong with our allocation?"
        </div>
    </div>
    """)
    return


# ─── ACT I: PREDICTION LOCK ───────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Your Prediction

    *Before running the calculator, commit to your hypothesis about what is causing the
    end-to-end SLO violation:*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    act1_pred = mo.ui.radio(
        options={
            "A) One service has a latency bug — the P99 measurements are incorrect": "option_a",
            "B) SLO violations compound probabilistically — 5 services each at P99 gives ~5% end-to-end violations": "option_b",
            "C) The 100ms-per-service allocation is too tight, each service needs more headroom": "option_c",
            "D) P99 is the wrong metric — the team should measure P50 instead": "option_d",
        },
        label="I predict the SLO violation is caused by:",
    )
    act1_pred
    return (act1_pred,)


@app.cell(hide_code=True)
def _(mo, act1_pred):
    mo.stop(
        act1_pred.value is None,
        mo.callout(mo.md("Select your prediction to continue to the SLO Composition Calculator."), kind="warn")
    )
    mo.callout(mo.md(f"**Prediction locked:** {act1_pred.value.split(')')[0]}. Now run the calculator to test your hypothesis."), kind="info")
    return


# ─── ACT I: INSTRUMENTS ───────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### SLO Composition Calculator

    Adjust the parameters to model the SRE Lead's production system. The calculator
    applies the tail latency composition formula to show the gap between per-service
    compliance and end-to-end compliance.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    n_services_slider = mo.ui.slider(
        start=2, stop=10, value=5, step=1,
        label="Number of services in critical path"
    )
    per_svc_violation_slider = mo.ui.slider(
        start=0.1, stop=5.0, value=1.0, step=0.1,
        label="Per-service SLO violation rate (%)"
    )
    correlation_slider = mo.ui.slider(
        start=0.0, stop=1.0, value=0.0, step=0.05,
        label="Service failure correlation (0=independent, 1=fully correlated)"
    )
    mo.vstack([
        mo.hstack([n_services_slider, per_svc_violation_slider], justify="center", gap="2rem"),
        mo.hstack([correlation_slider], justify="center"),
    ])
    return (n_services_slider, per_svc_violation_slider, correlation_slider)


@app.cell(hide_code=True)
def _(mo, n_services_slider, per_svc_violation_slider, correlation_slider, COLORS, apply_plotly_theme, go, np):
    _n = n_services_slider.value
    _p_svc = per_svc_violation_slider.value / 100.0   # fraction
    _rho = correlation_slider.value                    # correlation factor

    # ── SLO Composition Formula ────────────────────────────────────────────────
    # Independent case (lower bound):
    #   P(e2e violation) = 1 - (1 - p_svc)^n
    # Correlated case (interpolated):
    #   P(e2e violation) = (1 - rho) * P_independent + rho * p_svc
    #   At rho=1: correlated failures — e2e violation = per-service rate (best case)
    #   At rho=0: independent — violations multiply (worst case)
    _p_indep = 1.0 - (1.0 - _p_svc) ** _n
    _p_corr  = (1.0 - _rho) * _p_indep + _rho * _p_svc
    _p_e2e_pct = _p_corr * 100.0

    # The naive (wrong) assumption: linear addition
    _p_naive_pct = _p_svc * 100.0  # student's wrong mental model

    # Correct per-service budget: solve for p_i such that e2e violation = target
    _target_e2e = 0.01             # 1% end-to-end SLO violation budget
    _correct_per_svc_pct = (1.0 - (1.0 - _target_e2e) ** (1.0 / _n)) * 100.0

    # ── Color coding ──────────────────────────────────────────────────────────
    _color_e2e = (COLORS["GreenLine"] if _p_e2e_pct < 1.0
                  else COLORS["OrangeLine"] if _p_e2e_pct < 3.0
                  else COLORS["RedLine"])
    _color_svc = COLORS["GreenLine"] if _p_svc * 100 < 2.0 else COLORS["OrangeLine"]

    # ── Chart: violation probability vs number of services ────────────────────
    _x_range = list(range(2, 11))
    _y_indep  = [(1.0 - (1.0 - _p_svc) ** k) * 100 for k in _x_range]
    _y_corr   = [((1.0 - _rho) * (1.0 - (1.0 - _p_svc) ** k) + _rho * _p_svc) * 100
                 for k in _x_range]
    _y_naive  = [_p_svc * 100] * len(_x_range)   # wrong assumption: flat line

    _fig = go.Figure()
    _fig.add_trace(go.Scatter(
        x=_x_range, y=_y_indep,
        mode="lines+markers",
        name="Independent services",
        line=dict(color=COLORS["RedLine"], width=2.5),
        marker=dict(size=7),
    ))
    _fig.add_trace(go.Scatter(
        x=_x_range, y=_y_corr,
        mode="lines+markers",
        name=f"Correlated (ρ={_rho:.2f})",
        line=dict(color=COLORS["OrangeLine"], width=2.5, dash="dot"),
        marker=dict(size=7),
    ))
    _fig.add_trace(go.Scatter(
        x=_x_range, y=_y_naive,
        mode="lines",
        name="Wrong assumption (per-service rate only)",
        line=dict(color=COLORS["BlueLine"], width=1.5, dash="dash"),
    ))
    _fig.add_vline(x=_n, line_dash="dot", line_color="#94a3b8", line_width=1.5,
                   annotation_text=f"N={_n}", annotation_position="top right")
    _fig.add_hline(y=1.0, line_dash="dash", line_color=COLORS["GreenLine"], line_width=1.5,
                   annotation_text="1% e2e target", annotation_position="right")
    _fig.update_layout(
        title="End-to-End SLO Violation Rate vs. Service Chain Length",
        xaxis_title="Number of Services in Critical Path",
        yaxis_title="P(End-to-End SLO Violation) %",
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.9)"),
        height=340,
    )
    apply_plotly_theme(_fig)

    mo.vstack([
        mo.md("""
        #### Physics

        ```
        SLO Composition Formula (independence assumption):
          P(e2e violation) = 1 - Π(1 - P_i)
                           = 1 - (1 - p_svc)^n

        Correlated case (interpolated):
          P(e2e | ρ) = (1 - ρ) × P_independent + ρ × p_svc
        ```
        """),
        mo.Html(f"""
        <div style="display: flex; gap: 16px; justify-content: center;
                    flex-wrap: wrap; margin: 16px 0;">
            <div style="padding: 18px 22px; border: 1px solid #e2e8f0; border-radius: 10px;
                        min-width: 180px; text-align: center; background: white;
                        box-shadow: 0 2px 6px rgba(0,0,0,0.04);">
                <div style="color: #94a3b8; font-size: 0.82rem; font-weight: 600;
                            text-transform: uppercase; letter-spacing: 0.06em;">
                    Per-Service P99 Violation
                </div>
                <div style="font-size: 2rem; font-weight: 800; color: {_color_svc};
                            font-family: 'SF Mono', monospace; margin: 6px 0;">
                    {_p_svc * 100:.1f}%
                </div>
                <div style="color: #94a3b8; font-size: 0.78rem;">each of {_n} services</div>
            </div>
            <div style="padding: 18px 22px; border: 1px solid #e2e8f0; border-radius: 10px;
                        min-width: 180px; text-align: center; background: white;
                        box-shadow: 0 2px 6px rgba(0,0,0,0.04);">
                <div style="color: #94a3b8; font-size: 0.82rem; font-weight: 600;
                            text-transform: uppercase; letter-spacing: 0.06em;">
                    End-to-End Violation
                </div>
                <div style="font-size: 2rem; font-weight: 800; color: {_color_e2e};
                            font-family: 'SF Mono', monospace; margin: 6px 0;">
                    {_p_e2e_pct:.2f}%
                </div>
                <div style="color: #94a3b8; font-size: 0.78rem;">actual compound rate</div>
            </div>
            <div style="padding: 18px 22px; border: 1px solid #e2e8f0; border-radius: 10px;
                        min-width: 180px; text-align: center; background: white;
                        box-shadow: 0 2px 6px rgba(0,0,0,0.04);">
                <div style="color: #94a3b8; font-size: 0.82rem; font-weight: 600;
                            text-transform: uppercase; letter-spacing: 0.06em;">
                    Correct Per-Service Budget
                </div>
                <div style="font-size: 2rem; font-weight: 800; color: {COLORS['BlueLine']};
                            font-family: 'SF Mono', monospace; margin: 6px 0;">
                    {_correct_per_svc_pct:.3f}%
                </div>
                <div style="color: #94a3b8; font-size: 0.78rem;">for 1% e2e target</div>
            </div>
            <div style="padding: 18px 22px; border: 1px solid #e2e8f0; border-radius: 10px;
                        min-width: 180px; text-align: center; background: white;
                        box-shadow: 0 2px 6px rgba(0,0,0,0.04);">
                <div style="color: #94a3b8; font-size: 0.82rem; font-weight: 600;
                            text-transform: uppercase; letter-spacing: 0.06em;">
                    Allocation Gap
                </div>
                <div style="font-size: 2rem; font-weight: 800;
                            color: {COLORS['OrangeLine'] if _p_svc * 100 > _correct_per_svc_pct else COLORS['GreenLine']};
                            font-family: 'SF Mono', monospace; margin: 6px 0;">
                    {_p_svc * 100 / _correct_per_svc_pct:.1f}×
                </div>
                <div style="color: #94a3b8; font-size: 0.78rem;">over-allocated vs correct</div>
            </div>
        </div>
        """),
        mo.as_html(_fig),
    ])
    return (_p_e2e_pct, _correct_per_svc_pct, _p_svc, _n, _rho)


# ─── ACT I: PREDICTION vs REALITY OVERLAY ─────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, act1_pred, _p_e2e_pct, _p_svc):
    _correct_option = "option_b"
    _student_chose_correct = act1_pred.value == _correct_option

    _per_svc_display = _p_svc * 100
    _ratio = _p_e2e_pct / _per_svc_display if _per_svc_display > 0 else float("inf")

    if _student_chose_correct:
        _overlay = mo.callout(mo.md(
            f"**Prediction correct.** You predicted that SLO violations compound probabilistically. "
            f"With {round(1/_p_svc) if _p_svc < 1 else int(1/_p_svc)} — actually, with "
            f"per-service violation rate {_per_svc_display:.1f}%, the end-to-end violation rate is "
            f"**{_p_e2e_pct:.2f}%** — that is **{_ratio:.1f}×** the per-service rate. "
            f"The naive assumption that allocating 100ms × 5 services = 500ms end-to-end SLO is "
            f"wrong because violations at each service are *statistically independent events* "
            f"that compound via the tail composition formula."
        ), kind="success")
    elif act1_pred.value == "option_a":
        _overlay = mo.callout(mo.md(
            f"**Incorrect.** The per-service measurements are not wrong — all five services genuinely "
            f"meet their individual P99 SLOs. The issue is that meeting all per-service SLOs is "
            f"*necessary but not sufficient* for end-to-end compliance. With {_per_svc_display:.1f}% "
            f"per-service violation rate across 5 independent services, the end-to-end violation rate "
            f"is **{_p_e2e_pct:.2f}%** — {_ratio:.1f}× the per-service rate. "
            f"This is purely a consequence of tail latency composition, not a measurement bug."
        ), kind="warn")
    elif act1_pred.value == "option_c":
        _overlay = mo.callout(mo.md(
            f"**Incorrect.** The per-service SLO values (100ms each) are not the problem — "
            f"the *allocation method* is. Even if each service had P99 = 50ms with 0.5% violation rate, "
            f"the end-to-end violation would still be "
            f"{(1.0 - (1.0 - 0.005)**5)*100:.2f}% across 5 services. "
            f"The fundamental error is treating end-to-end budget as a simple sum of per-service budgets. "
            f"The tail composition formula shows the actual end-to-end violation is **{_p_e2e_pct:.2f}%** "
            f"— {_ratio:.1f}× the per-service rate — regardless of how tight the individual SLOs are set."
        ), kind="warn")
    else:
        _overlay = mo.callout(mo.md(
            f"**Incorrect.** P99 is precisely the right metric for tail latency SLOs — it captures "
            f"the worst-case experience for the top 1% of requests, which is exactly where SLA "
            f"violations occur. The issue is compositional: with {_per_svc_display:.1f}% per-service "
            f"violation rate across 5 services, the end-to-end violation is **{_p_e2e_pct:.2f}%** "
            f"({_ratio:.1f}× the per-service rate). Switching to P50 would *hide* the problem, "
            f"not fix it."
        ), kind="warn")

    mo.vstack([
        mo.md("### Prediction vs. Reality"),
        _overlay,
    ])
    return


# ─── ACT I: REFLECTION ────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Reflection

    *Now that you have seen the composition mechanics, answer the structural question:*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    act1_reflect = mo.ui.radio(
        options={
            "A) Divide total budget equally: allocate SLO/N to each service": "reflect_a",
            "B) Error budget allocation: tighter SLOs for slower/less reliable services, looser for fast ones — solve P(e2e) = target for per-service budget": "reflect_b",
            "C) Give each service 10x headroom to guarantee the end-to-end SLO": "reflect_c",
            "D) Monitor only the end-to-end SLO — per-service SLOs cause false confidence": "reflect_d",
        },
        label="The correct approach to per-service SLO allocation is:",
    )
    act1_reflect
    return (act1_reflect,)


@app.cell(hide_code=True)
def _(mo, act1_reflect):
    mo.stop(
        act1_reflect.value is None,
        mo.callout(mo.md("Select your answer to see the explanation."), kind="warn")
    )

    if act1_reflect.value == "reflect_b":
        mo.callout(mo.md(
            "**Correct.** Error budget allocation solves for the per-service violation rate that "
            "achieves the target end-to-end SLO: `p_svc = 1 - (1 - p_e2e_target)^(1/N)`. "
            "For a 1% end-to-end target across 5 services, the correct per-service budget is "
            "`1 - 0.99^(1/5) ≈ 0.201%` — roughly 5× tighter than the naive 1%/5=0.2% "
            "(which is coincidentally close, but the formula is not a simple division). "
            "Critically, budget should be *unequal*: slower or more complex services "
            "(e.g., model inference) receive tighter SLOs because their tail latency naturally "
            "has higher variance and dominates end-to-end violations."
        ), kind="success")
    elif act1_reflect.value == "reflect_a":
        mo.callout(mo.md(
            "**Not quite.** Equal division (SLO/N) is exactly the mistake the SRE Lead made. "
            "100ms × 5 = 500ms seems correct, but it conflates *latency budget* (how long each "
            "service can take) with *violation rate budget* (what fraction of requests can miss "
            "the SLO). The correct approach is to solve the composition formula: "
            "`p_svc = 1 - (1 - p_e2e_target)^(1/N)`. Equal division of the violation budget "
            "also ignores that some services (inference, feature store) have higher inherent "
            "latency variance and require tighter SLOs."
        ), kind="warn")
    elif act1_reflect.value == "reflect_c":
        mo.callout(mo.md(
            "**Not quite.** 10× headroom is an arbitrary safety factor that does not derive "
            "from the composition formula. It would make per-service SLOs far too aggressive, "
            "driving unnecessary engineering cost and potentially masking real performance "
            "regressions. The principled approach is to use `p_svc = 1 - (1 - p_e2e)^(1/N)`, "
            "which gives the exact headroom needed — no more, no less."
        ), kind="warn")
    else:
        mo.callout(mo.md(
            "**Not quite.** Eliminating per-service SLOs makes root cause analysis impossible. "
            "When the end-to-end SLO is violated, you need per-service metrics to identify "
            "*which* service caused the violation. The correct approach is to maintain both: "
            "per-service SLOs derived from the composition formula, plus the end-to-end SLO "
            "as the customer-facing commitment. The per-service dashboard stays green only when "
            "the allocation math is correct."
        ), kind="warn")
    return


# ─── ACT I: MATHPEEK ──────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "The governing equation — SLO composition": mo.md("""
        **SLO Composition Formula (independent services):**

        $$P(\\text{e2e violation}) = 1 - \\prod_{i=1}^{N} (1 - P_i)$$

        where $P_i$ is the per-service SLO violation rate (fraction).

        **For identical services at rate $p$:**

        $$P(\\text{e2e}) = 1 - (1 - p)^N$$

        **Error budget allocation (solve for per-service budget):**

        $$p_{\\text{svc}} = 1 - (1 - p_{\\text{e2e target}})^{1/N}$$

        **Example:** 5-service chain, 1% e2e target:
        $$p_{\\text{svc}} = 1 - 0.99^{0.2} \\approx 0.201\\%$$

        **Correlated failure interpolation:**

        $$P(\\text{e2e} | \\rho) = (1-\\rho) \\cdot P_{\\text{independent}} + \\rho \\cdot p_{\\text{svc}}$$

        At $\\rho = 0$: fully independent (worst case — violations compound maximally).
        At $\\rho = 1$: fully correlated (best case — e2e violation equals per-service rate).

        In practice, services sharing infrastructure (same cluster, same database) exhibit
        partial correlation — and that correlation *reduces* the e2e violation rate only if
        failures are synchronous. Correlated failures in *different* services driven by the
        same root cause (e.g., network partition) can create dependency amplification instead.

        - **$P_i$** — per-service SLO violation rate
        - **$N$** — number of services in critical path
        - **$\\rho$** — correlation coefficient (0 = independent, 1 = fully correlated)
        - **$p_{\\text{e2e target}}$** — desired end-to-end violation budget
        """)
    })
    return


# ═════════════════════════════════════════════════════════════════════════════
# ACT II: CASCADING FAILURE PREVENTION
# Stakeholder: Platform Reliability Lead | Prediction: cascade prevention
# ═════════════════════════════════════════════════════════════════════════════

@app.cell(hide_code=True)
def _(mo):
    mo.Html("""
    <div style="margin: 36px 0 8px 0;">
        <div style="font-size: 0.72rem; font-weight: 700; letter-spacing: 0.14em;
                    text-transform: uppercase; color: #94a3b8; margin-bottom: 6px;
                    display: flex; align-items: center; gap: 8px;">
            <span style="background: #CB202D; color: white; border-radius: 50%;
                         width: 22px; height: 22px; display: inline-flex; align-items: center;
                         justify-content: center; font-size: 0.72rem; font-weight: 800;">II</span>
            Act II — Design Challenge · 20–25 min
            <span style="flex: 1; height: 1px; background: #e2e8f0;"></span>
        </div>
        <div style="font-size: 1.5rem; font-weight: 800; color: #0f172a; line-height: 1.2;">
            Cascading Failure Prevention
        </div>
        <div style="font-size: 0.95rem; color: #475569; margin-top: 4px;">
            A slow service fills queues. Full queues cause timeouts. Timeouts trigger retries.
            Retries amplify load. Amplified load kills databases. The entire system goes dark.
        </div>
    </div>
    """)
    return


# ─── ACT II: STAKEHOLDER MESSAGE ──────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, COLORS):
    _color = COLORS["RedLine"]
    mo.Html(f"""
    <div style="border-left: 4px solid {_color}; background: {COLORS['RedLL']};
                border-radius: 0 10px 10px 0; padding: 16px 22px; margin: 12px 0;">
        <div style="font-size: 0.72rem; font-weight: 700; color: {_color};
                    text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px;">
            Incoming Message · Platform Reliability Lead
        </div>
        <div style="font-style: italic; font-size: 1.0rem; color: #1e293b; line-height: 1.65;">
            "We had a 6-hour total outage last Tuesday. Post-mortem shows: slow model inference
            at 180% capacity → request queue in the API gateway built up to 50,000 requests →
            clients hit their 2-second timeouts → they retried 3 times each → the database
            handling request state received 3× normal load → database query latency spiked →
            all upstream services started timing out → entire platform unavailable. How do we
            architect to prevent this? We are on Kubernetes and our HPA took 30+ seconds to
            scale — by then we were already cascading."
        </div>
    </div>
    """)
    return


# ─── ACT II: PREDICTION LOCK ──────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Your Prediction

    *Before running the cascade simulator, commit to your hypothesis about the
    best approach to prevent this cascade:*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    act2_pred = mo.ui.radio(
        options={
            "A) Add more capacity — cascades only happen on undersized systems": "option_a",
            "B) Circuit breakers and bulkheads: isolate failures, shed excess load rather than queue it, prevent retry storms": "option_b",
            "C) Better monitoring — detect the cascade early and intervene manually before it propagates": "option_c",
            "D) Use Kubernetes HPA — auto-scaling prevents cascade failures automatically": "option_d",
        },
        label="The best approach to prevent cascading failures is:",
    )
    act2_pred
    return (act2_pred,)


@app.cell(hide_code=True)
def _(mo, act2_pred):
    mo.stop(
        act2_pred.value is None,
        mo.callout(mo.md("Select your prediction to continue to the cascade simulator."), kind="warn")
    )
    mo.callout(mo.md(f"**Prediction locked:** {act2_pred.value.split(')')[0]}. Run the simulator to test your prediction."), kind="info")
    return


# ─── ACT II: INSTRUMENTS ──────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Cascade Failure Simulator

    Configure the system parameters and failure mitigations. The simulator models
    60 seconds of real-time cascade dynamics across a 5-service ML serving platform.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    overload_pct_slider = mo.ui.slider(
        start=110, stop=200, value=150, step=5,
        label="Initial overload (% of service capacity)"
    )
    timeout_slider = mo.ui.slider(
        start=100, stop=10000, value=2000, step=100,
        label="Client timeout (ms)"
    )
    retry_multiplier_slider = mo.ui.slider(
        start=1.0, stop=5.0, value=3.0, step=0.5,
        label="Retry multiplier (retries per timeout)"
    )
    cb_threshold_slider = mo.ui.slider(
        start=1, stop=20, value=5, step=1,
        label="Circuit breaker threshold (failures before opening)"
    )
    circuit_breakers_toggle = mo.ui.radio(
        options={"Circuit breakers: OFF": False, "Circuit breakers: ON": True},
        value="Circuit breakers: OFF",
        label="Circuit Breakers",
        inline=True,
    )
    bulkheads_toggle = mo.ui.radio(
        options={"Bulkheads: OFF": False, "Bulkheads: ON": True},
        value="Bulkheads: OFF",
        label="Bulkheads (service isolation)",
        inline=True,
    )
    mo.vstack([
        mo.hstack([overload_pct_slider, timeout_slider], justify="center", gap="2rem"),
        mo.hstack([retry_multiplier_slider, cb_threshold_slider], justify="center", gap="2rem"),
        mo.md("---"),
        mo.hstack([circuit_breakers_toggle, bulkheads_toggle], justify="center", gap="3rem"),
    ])
    return (
        overload_pct_slider, timeout_slider, retry_multiplier_slider,
        cb_threshold_slider, circuit_breakers_toggle, bulkheads_toggle,
    )


@app.cell(hide_code=True)
def _(
    mo, COLORS, apply_plotly_theme, go, np,
    context_toggle,
    overload_pct_slider, timeout_slider, retry_multiplier_slider,
    cb_threshold_slider, circuit_breakers_toggle, bulkheads_toggle,
    K8S_SCALE_LAG_SEC,
):
    # ── Simulation parameters ─────────────────────────────────────────────────
    _overload    = overload_pct_slider.value / 100.0   # 1.5 = 150% capacity
    _timeout_ms  = timeout_slider.value                 # ms
    _retry_mult  = retry_multiplier_slider.value        # retries on timeout
    _cb_thresh   = cb_threshold_slider.value            # failures to open CB
    _cb_on       = circuit_breakers_toggle.value
    _bh_on       = bulkheads_toggle.value
    _is_k8s      = context_toggle.value == "kubernetes"
    _n_svc       = 5                                    # services: gateway, auth, features, inference, formatter

    # ── Cascade simulation model ───────────────────────────────────────────────
    # State: utilization of each service over time (60 simulated seconds)
    # Physics:
    #   1. Overloaded service fills queue → requests exceed timeout → retries
    #   2. Retry load propagates upstream (retry storm)
    #   3. Queue depth grows as: Q(t+1) = Q(t) + arrival_rate - service_rate
    #   4. Circuit breaker opens after cb_thresh consecutive failures → sheds load
    #   5. Bulkheads prevent cross-service queue contamination
    #   6. K8s HPA delay: scaling kicks in after K8S_SCALE_LAG_SEC seconds

    _T           = 60   # simulation duration in seconds
    _dt          = 0.5  # time step (seconds)
    _steps       = int(_T / _dt)
    _times       = [i * _dt for i in range(_steps)]

    # Service names (ordered: inference is service 3, most constrained)
    _svc_names   = ["API Gateway", "Auth Service", "Feature Store", "ML Inference", "Formatter"]
    _svc_baseline_util = [0.40, 0.35, 0.55, 0.80, 0.25]  # baseline utilization

    # Initialize utilization arrays
    _svc_util    = [[u] for u in _svc_baseline_util]
    _cb_state    = [False] * _n_svc       # circuit breaker open state
    _cb_counter  = [0] * _n_svc          # consecutive failure counter
    _queue_depth = [0.0] * _n_svc        # request queue depth (normalized)

    # Simulate cascade
    for _step in range(1, _steps):
        _t = _step * _dt
        _new_utils = []

        for _svc_idx in range(_n_svc):
            _prev_util = _svc_util[_svc_idx][-1]

            # Initial overload hits inference service (index 3) at t=0
            if _svc_idx == 3:
                _base_load = _overload
            else:
                _base_load = _svc_baseline_util[_svc_idx]

            # Retry storm amplification: requests that time out come back multiplied
            # Timeout occurs when util > 0.95 (queue fills, latency spikes)
            _is_timing_out = _prev_util > 0.95
            _retry_amplification = _retry_mult if _is_timing_out and not _cb_state[_svc_idx] else 1.0

            # Load from downstream retry storms (upstream services get amplified load)
            # When inference overloads, all upstream services (gateway, auth, features) see retries
            _upstream_cascade = 0.0
            if _svc_idx < 3:
                _inf_util = _svc_util[3][-1] if len(_svc_util[3]) > 0 else _svc_baseline_util[3]
                if _inf_util > 0.95:
                    # Timeout → retry amplification cascades upstream
                    _cascade_factor = (_retry_mult - 1.0) * 0.3 if not _bh_on else 0.0
                    _upstream_cascade = _cascade_factor * (_inf_util - 0.95) * 2.0

            # Kubernetes HPA response: scaling reduces load after lag
            _k8s_reduction = 0.0
            if _is_k8s and _t > K8S_SCALE_LAG_SEC and _base_load > 1.0:
                # HPA has kicked in, starts reducing overload
                _k8s_reduction = min(0.3, (_t - K8S_SCALE_LAG_SEC) * 0.01)

            # Circuit breaker: if open, shed load immediately
            _cb_shed = 0.0
            if _cb_on:
                if _prev_util > 0.95:
                    _cb_counter[_svc_idx] += 1
                else:
                    _cb_counter[_svc_idx] = max(0, _cb_counter[_svc_idx] - 1)

                if _cb_counter[_svc_idx] >= _cb_thresh:
                    _cb_state[_svc_idx] = True
                elif _prev_util < 0.7:
                    # Half-open: probe recovery
                    _cb_state[_svc_idx] = False
                    _cb_counter[_svc_idx] = 0

                if _cb_state[_svc_idx]:
                    _cb_shed = (_base_load - 1.0) * 0.8 if _base_load > 1.0 else 0.0

            # Compute new utilization
            _new_util = (_base_load * _retry_amplification
                         + _upstream_cascade
                         - _k8s_reduction
                         - _cb_shed)
            _new_util = max(0.05, min(2.0, _new_util))
            _new_utils.append(_new_util)

        for _idx in range(_n_svc):
            _svc_util[_idx].append(_new_utils[_idx])

    # ── Compute cascade severity metrics ──────────────────────────────────────
    _final_utils       = [_svc_util[i][-1] for i in range(_n_svc)]
    _peak_utils        = [max(_svc_util[i]) for i in range(_n_svc)]
    _services_overloaded = sum(1 for u in _final_utils if u > 1.0)
    _cascade_contained  = _services_overloaded <= 2
    _recovery_minutes   = max(_final_utils) * 15 if not _cascade_contained else 2.0
    _is_full_outage     = _services_overloaded >= _n_svc

    # ── Build Plotly chart ────────────────────────────────────────────────────
    _svc_colors = [
        COLORS["BlueLine"],
        COLORS["GreenLine"],
        COLORS["OrangeLine"],
        COLORS["RedLine"],
        "#8b5cf6",   # purple for formatter
    ]

    _fig2 = go.Figure()
    for _si in range(_n_svc):
        _fig2.add_trace(go.Scatter(
            x=_times,
            y=_svc_util[_si],
            mode="lines",
            name=_svc_names[_si],
            line=dict(color=_svc_colors[_si], width=2.0),
        ))

    # Danger zone shading
    _fig2.add_hrect(y0=1.0, y1=2.0, fillcolor="rgba(203,32,45,0.06)", line_width=0)
    _fig2.add_hline(y=1.0, line_dash="dash", line_color=COLORS["RedLine"], line_width=1.5,
                    annotation_text="Capacity = 100%", annotation_position="right")

    # K8s HPA response marker
    if context_toggle.value == "kubernetes":
        _fig2.add_vline(x=K8S_SCALE_LAG_SEC, line_dash="dot",
                        line_color=COLORS["BlueLine"], line_width=1.5,
                        annotation_text=f"HPA kicks in ({K8S_SCALE_LAG_SEC}s)",
                        annotation_position="top right")

    _cb_label = f"CB={'ON' if _cb_on else 'OFF'}, BH={'ON' if _bh_on else 'OFF'}"
    _fig2.update_layout(
        title=f"Cascade Simulation — {_cb_label} — {_services_overloaded}/{_n_svc} services overloaded",
        xaxis_title="Simulation time (seconds)",
        yaxis_title="Service utilization (1.0 = capacity)",
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.9)"),
        height=360,
    )
    apply_plotly_theme(_fig2)

    # ── Metrics row ───────────────────────────────────────────────────────────
    _color_svc_ol = (COLORS["GreenLine"] if _services_overloaded == 0
                     else COLORS["OrangeLine"] if _services_overloaded <= 2
                     else COLORS["RedLine"])
    _color_recovery = (COLORS["GreenLine"] if _recovery_minutes < 5
                       else COLORS["OrangeLine"] if _recovery_minutes < 20
                       else COLORS["RedLine"])

    mo.vstack([
        mo.md("""
        #### Cascade Physics

        ```
        Queue buildup:   Q(t+1) = Q(t) + arrival_rate - service_rate
        Retry storm:     effective_load = base_load × retry_multiplier  (when queue overflows)
        Circuit breaker: open when consecutive_failures ≥ cb_threshold
                         shed_load = (utilization - 1.0) × 0.8
        Bulkhead:        upstream_cascade = 0 when bulkheads ON
        K8s HPA:         scale_down_starts at t = 30s (typical response time)
        ```
        """),
        mo.Html(f"""
        <div style="display: flex; gap: 16px; justify-content: center;
                    flex-wrap: wrap; margin: 12px 0;">
            <div style="padding: 18px 22px; border: 1px solid #e2e8f0; border-radius: 10px;
                        min-width: 168px; text-align: center; background: white;
                        box-shadow: 0 2px 6px rgba(0,0,0,0.04);">
                <div style="color: #94a3b8; font-size: 0.82rem; font-weight: 600;
                            text-transform: uppercase; letter-spacing: 0.06em;">
                    Services Overloaded
                </div>
                <div style="font-size: 2rem; font-weight: 800; color: {_color_svc_ol};
                            font-family: 'SF Mono', monospace; margin: 6px 0;">
                    {_services_overloaded}/{_n_svc}
                </div>
                <div style="color: #94a3b8; font-size: 0.78rem;">at t=60s</div>
            </div>
            <div style="padding: 18px 22px; border: 1px solid #e2e8f0; border-radius: 10px;
                        min-width: 168px; text-align: center; background: white;
                        box-shadow: 0 2px 6px rgba(0,0,0,0.04);">
                <div style="color: #94a3b8; font-size: 0.82rem; font-weight: 600;
                            text-transform: uppercase; letter-spacing: 0.06em;">
                    Cascade Contained
                </div>
                <div style="font-size: 2rem; font-weight: 800;
                            color: {COLORS['GreenLine'] if _cascade_contained else COLORS['RedLine']};
                            font-family: 'SF Mono', monospace; margin: 6px 0;">
                    {'YES' if _cascade_contained else 'NO'}
                </div>
                <div style="color: #94a3b8; font-size: 0.78rem;">
                    {'fault isolated' if _cascade_contained else 'propagated'}
                </div>
            </div>
            <div style="padding: 18px 22px; border: 1px solid #e2e8f0; border-radius: 10px;
                        min-width: 168px; text-align: center; background: white;
                        box-shadow: 0 2px 6px rgba(0,0,0,0.04);">
                <div style="color: #94a3b8; font-size: 0.82rem; font-weight: 600;
                            text-transform: uppercase; letter-spacing: 0.06em;">
                    Est. Recovery Time
                </div>
                <div style="font-size: 2rem; font-weight: 800; color: {_color_recovery};
                            font-family: 'SF Mono', monospace; margin: 6px 0;">
                    {_recovery_minutes:.0f}m
                </div>
                <div style="color: #94a3b8; font-size: 0.78rem;">after intervention</div>
            </div>
            <div style="padding: 18px 22px; border: 1px solid #e2e8f0; border-radius: 10px;
                        min-width: 168px; text-align: center; background: white;
                        box-shadow: 0 2px 6px rgba(0,0,0,0.04);">
                <div style="color: #94a3b8; font-size: 0.82rem; font-weight: 600;
                            text-transform: uppercase; letter-spacing: 0.06em;">
                    Retry Amplification
                </div>
                <div style="font-size: 2rem; font-weight: 800; color: {COLORS['OrangeLine']};
                            font-family: 'SF Mono', monospace; margin: 6px 0;">
                    {_retry_mult:.1f}×
                </div>
                <div style="color: #94a3b8; font-size: 0.78rem;">effective load multiplier</div>
            </div>
        </div>
        """),
        mo.as_html(_fig2),
    ])
    return (_services_overloaded, _n_svc, _recovery_minutes, _cascade_contained, _is_full_outage)


# ─── ACT II: FAILURE STATE (DANGER CALLOUT) ───────────────────────────────────
@app.cell(hide_code=True)
def _(mo, _is_full_outage, _n_svc, _recovery_minutes):
    if _is_full_outage:
        mo.callout(mo.md(
            f"**CASCADE FAILURE: All {_n_svc} services overloaded.** "
            f"Recovery time: ~{_recovery_minutes:.0f} minutes. "
            f"The retry storm amplified initial overload into full system collapse. "
            f"Enable circuit breakers to contain the fault to 1-2 services. "
            f"Enable bulkheads to prevent upstream contamination from downstream failures."
        ), kind="danger")
    else:
        mo.md("")
    return


# ─── ACT II: PREDICTION FEEDBACK ──────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, act2_pred, _cascade_contained, _services_overloaded, _n_svc, _recovery_minutes, K8S_SCALE_LAG_SEC):
    _correct_option = "option_b"
    _chose_correct = act2_pred.value == _correct_option

    if _chose_correct:
        _feedback = mo.callout(mo.md(
            f"**Prediction correct.** Circuit breakers and bulkheads are the correct architectural "
            f"response. The simulator shows: with circuit breakers and bulkheads ON, the cascade "
            f"is contained to {_services_overloaded}/{_n_svc} services with ~{_recovery_minutes:.0f}-minute "
            f"recovery. Without them, the retry storm amplifies initial overload into full system collapse. "
            f"The key mechanism: circuit breakers *shed* excess load immediately (no queuing), "
            f"and bulkheads prevent queue contamination from crossing service boundaries."
        ), kind="success")
    elif act2_pred.value == "option_a":
        _feedback = mo.callout(mo.md(
            f"**Incorrect.** Capacity is not the constraint — cascade dynamics are. Even at 2× "
            f"the current capacity, the retry storm multiplier ({_services_overloaded} services "
            f"affected) means a sufficiently large traffic spike will eventually exhaust any "
            f"finite capacity and trigger the same cascade. The failure mode is architectural: "
            f"queued requests consume memory and processing time for work that will timeout "
            f"anyway, while retries add more load than the shed removes. Circuit breakers "
            f"break this amplification loop regardless of capacity."
        ), kind="warn")
    elif act2_pred.value == "option_c":
        _feedback = mo.callout(mo.md(
            f"**Incorrect.** The simulator shows full cascade completes in under 30 seconds. "
            f"Human intervention takes minutes — the system is fully down before any on-call "
            f"engineer can respond. Cascades operate at machine speed, not human speed. "
            f"Better monitoring is valuable for post-mortem analysis and detecting early warning "
            f"signals, but it cannot substitute for automated circuit breakers that respond in "
            f"under {100}ms — three orders of magnitude faster than human response."
        ), kind="warn")
    else:
        _feedback = mo.callout(mo.md(
            f"**Incorrect.** Kubernetes HPA has a {K8S_SCALE_LAG_SEC}-second response lag (the "
            f"time to detect overload, schedule pods, pull images, and start serving). The "
            f"cascade completes in under 30 seconds. The simulator shows the HPA marker at "
            f"t={K8S_SCALE_LAG_SEC}s — by that point, the cascade has already propagated to "
            f"multiple services. Auto-scaling addresses *sustained* overload over minutes; "
            f"it cannot respond to the seconds-scale cascade dynamics that circuit breakers handle."
        ), kind="warn")

    _feedback
    return


# ─── ACT II: REFLECTION ───────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Reflection

    *The simulator shows that load shedding (dropping requests) prevents cascades
    better than queuing them. Why?*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    act2_reflect = mo.ui.radio(
        options={
            "A) Dropping requests reduces server load — fewer requests, less compute": "reflect2_a",
            "B) Queued requests consume memory and processing time for work that will timeout anyway — the queue is a cascade amplifier": "reflect2_b",
            "C) Kubernetes auto-scaling compensates for dropped requests by adding replicas": "reflect2_c",
            "D) Dropped requests trigger alerts, enabling faster on-call response": "reflect2_d",
        },
        label="Load shedding prevents cascades better than queuing because:",
    )
    act2_reflect
    return (act2_reflect,)


@app.cell(hide_code=True)
def _(mo, act2_reflect):
    mo.stop(
        act2_reflect.value is None,
        mo.callout(mo.md("Select your answer to see the explanation."), kind="warn")
    )

    if act2_reflect.value == "reflect2_b":
        mo.callout(mo.md(
            "**Correct.** A full queue is a cascade amplifier, not a buffer. Every queued request "
            "that will eventually timeout: (1) consumes memory for the queue entry, (2) consumes "
            "CPU time when it is eventually processed (often only to return a timeout error), and "
            "(3) causes a retry when it does timeout, adding `retry_multiplier` new requests back "
            "into the system. The queue *increases* the effective load on an already-overloaded "
            "system. Load shedding breaks this loop: by immediately returning HTTP 429 (Too Many "
            "Requests), the server processes no work for that request, the client does not retry "
            "(if properly implemented), and the queue remains short. Short queues = short latencies "
            "= fewer timeouts = no retry storms."
        ), kind="success")
    elif act2_reflect.value == "reflect2_a":
        mo.callout(mo.md(
            "**Not quite.** The framing is correct but incomplete. Dropping requests does reduce "
            "server load, but the mechanism that prevents cascades is more specific: it prevents "
            "the *queue from acting as a cascade amplifier*. The critical insight is that queued "
            "requests that eventually timeout generate retries, which add more load than the "
            "original request. A queue that holds 10,000 requests, each of which times out and "
            "retries 3×, effectively becomes 40,000 requests — 4× the original load. Load "
            "shedding breaks this multiplication by refusing requests that would timeout anyway."
        ), kind="warn")
    elif act2_reflect.value == "reflect2_c":
        mo.callout(mo.md(
            "**Not quite.** Kubernetes HPA scaling takes 30+ seconds to respond, while the "
            "cascade completes in under 30 seconds. New replicas also need time to warm up "
            "and start serving. Critically, scaling adds capacity but does not address the "
            "retry storm already in flight — if 50,000 queued requests have already timed out "
            "and are being retried, adding more replicas simply gives the retry storm more "
            "targets to overwhelm. Circuit breakers must shed the retry load *before* scaling "
            "can help."
        ), kind="warn")
    else:
        mo.callout(mo.md(
            "**Not quite.** Alert-triggered human response operates on timescales of minutes, "
            "while cascades complete in seconds. The mechanism preventing the cascade is "
            "automated and immediate: circuit breakers respond in under 100ms by detecting "
            "consecutive failures and opening to shed load. The key insight is that the queue "
            "itself amplifies load through retry storms — shedding excess load before it enters "
            "the queue prevents the amplification from occurring, regardless of how fast alerts fire."
        ), kind="warn")
    return


# ─── ACT II: MATHPEEK ─────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, K8S_SCALE_LAG_SEC, K8S_POD_START_SEC, CIRCUIT_BREAKER_MS):
    _detect_sec = K8S_SCALE_LAG_SEC - K8S_POD_START_SEC
    _mathpeek_text = (
        "**Queue buildup model (M/M/1 queue approximation):**\n\n"
        "$$Q(t+1) = Q(t) + \\lambda - \\mu$$\n\n"
        "where $\\lambda$ = arrival rate, $\\mu$ = service rate. "
        "When $\\lambda > \\mu$: queue grows linearly, latency diverges.\n\n"
        "**Retry storm amplification:**\n\n"
        "$$L_{\\text{eff}}(t) = \\lambda_0 \\cdot r^{\\lfloor t / T_{\\text{timeout}} \\rfloor}$$\n\n"
        "where $r$ = retry multiplier, $T_{\\text{timeout}}$ = client timeout period. "
        "Each timeout wave amplifies load by $r\\times$ — exponential growth without circuit breakers.\n\n"
        "**Circuit breaker state machine:**\n\n"
        "```\n"
        f"CLOSED → OPEN  : consecutive_failures ≥ cb_threshold (HPA lag = {K8S_SCALE_LAG_SEC}s is separate)\n"
        "OPEN   → HALF  : after recovery_timeout (30–60s typical)\n"
        "HALF   → CLOSED: probe request succeeds\n"
        "HALF   → OPEN  : probe request fails\n"
        "```\n\n"
        f"Circuit breaker response time: **{CIRCUIT_BREAKER_MS} ms** (application-level detection).\n\n"
        "**Kubernetes HPA response time model:**\n\n"
        "$$T_{\\text{scale}} = T_{\\text{detect}} + T_{\\text{schedule}} + T_{\\text{pod\\_start}}$$\n\n"
        f"$$= {_detect_sec}\\,\\text{{s}} + \\epsilon + {K8S_POD_START_SEC}\\,\\text{{s}}"
        f"\\approx {K8S_SCALE_LAG_SEC}\\,\\text{{s}}$$\n\n"
        f"HPA polls metrics every 15s by default (Kubernetes HPA spec); pod startup adds "
        f"{K8S_POD_START_SEC}s for warm image. Total: **{K8S_SCALE_LAG_SEC}s** — 300× slower than circuit breakers.\n\n"
        "**Load shed effectiveness:**\n\n"
        "$$L_{\\text{shed}} = (\\rho - 1.0) \\times 0.8 \\quad \\text{when CB open}$$\n\n"
        "Bulkhead isolation prevents upstream cascade:\n\n"
        "$$L_{\\text{upstream\\_cascade}} = 0 \\quad \\text{when bulkheads enabled}$$\n\n"
        "- **$Q(t)$** — queue depth (normalized to service capacity)\n"
        "- **$\\lambda_0$** — baseline arrival rate\n"
        "- **$r$** — retry multiplier per timeout cycle\n"
        "- **$T_{\\text{timeout}}$** — client timeout period (ms)\n"
        "- **$\\rho$** — service utilization (>1.0 = overloaded)\n"
    )
    mo.accordion({
        "The governing equations — cascade dynamics": mo.md(_mathpeek_text)
    })
    return


# ─── LEDGER SAVE + HUD (hide_code=True) ───────────────────────────────────────
@app.cell(hide_code=True)
def _(
    mo, ledger, COLORS,
    context_toggle,
    act1_pred, act2_pred,
    _p_e2e_pct, _correct_per_svc_pct,
    _services_overloaded, _n_svc,
    _cascade_contained,
    circuit_breakers_toggle, bulkheads_toggle,
):
    _ctx          = context_toggle.value
    _cb_enabled   = circuit_breakers_toggle.value
    _bh_enabled   = bulkheads_toggle.value
    _act1_correct = act1_pred.value == "option_b"
    _act2_correct = act2_pred.value is not None

    ledger.save(
        chapter="v2_12",
        design={
            "context":                 _ctx,
            "circuit_breakers_enabled": _cb_enabled,
            "bulkheads_enabled":        _bh_enabled,
            "per_service_slo_ms":       100.0,         # SRE Lead's allocation
            "end_to_end_violation_pct": round(_p_e2e_pct, 3),
            "cascade_contained":        _cascade_contained,
            "act1_prediction":          act1_pred.value or "none",
            "act1_correct":             _act1_correct,
            "act2_result":              float(_services_overloaded),
            "act2_decision":            act2_pred.value or "none",
            "constraint_hit":           _services_overloaded >= _n_svc,
        }
    )

    _act1_icon  = "green"  if _act1_correct else "orange"
    _act2_icon  = "green"  if _cascade_contained else "red"
    _ctx_label  = "Kubernetes" if _ctx == "kubernetes" else "Bare Metal"
    _cb_label   = "ON" if _cb_enabled else "OFF"
    _bh_label   = "ON" if _bh_enabled else "OFF"

    mo.Html(f"""
    <div class="lab-hud">
        <div>
            <span class="hud-label">LAB</span>&nbsp;
            <span class="hud-value">V2-12 · Ops at Scale</span>
        </div>
        <div>
            <span class="hud-label">CONTEXT</span>&nbsp;
            <span class="hud-value">{_ctx_label}</span>
        </div>
        <div>
            <span class="hud-label">ACT I</span>&nbsp;
            <span class="{'hud-active' if _act1_correct else 'hud-none'}">
                {'CORRECT' if _act1_correct else 'INCORRECT'}
            </span>
        </div>
        <div>
            <span class="hud-label">E2E VIOLATION</span>&nbsp;
            <span class="hud-value">{_p_e2e_pct:.2f}%</span>
        </div>
        <div>
            <span class="hud-label">CIRCUIT BREAKERS</span>&nbsp;
            <span class="{'hud-active' if _cb_enabled else 'hud-none'}">{_cb_label}</span>
        </div>
        <div>
            <span class="hud-label">CASCADE</span>&nbsp;
            <span class="{'hud-active' if _cascade_contained else 'hud-none'}">
                {'CONTAINED' if _cascade_contained else 'FULL OUTAGE'}
            </span>
        </div>
        <div>
            <span class="hud-label">LEDGER</span>&nbsp;
            <span class="hud-active">SAVED</span>
        </div>
    </div>
    """)
    return


if __name__ == "__main__":
    app.run()

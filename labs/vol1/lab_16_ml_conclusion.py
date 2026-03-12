import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full")

# ─────────────────────────────────────────────────────────────────────────────
# LAB 16: THE ARCHITECT'S AUDIT
#
# Chapter: conclusion.qmd
# Core Invariant: Synthesis — all 7 invariants, cross-lab Design Ledger review
# Contexts: ALL 4 (Cloud H100, Edge Jetson Orin NX, Mobile Smartphone NPU, TinyML Cortex-M7)
#
# This is the capstone lab for Volume I. It does two things that no prior lab
# does individually:
#
#   1. It reads the student's entire Design Ledger from Labs 01–15 and renders
#      a personalized Mental Model Report Card — revealing which invariants
#      they consistently underestimated, which constraints they hit most often,
#      and what their dominant deployment intuition is.
#
#   2. It presents a Full-Stack Design Challenge: architect a 7B-parameter LLM
#      inference system that must satisfy simultaneous SLOs across all 4 tiers.
#      All prior invariants appear as active constraints. Every design decision
#      the student makes is cross-validated against the physics they have studied.
#
# Act I  — Design Ledger Archaeology (12–15 min)
#   Render complete Mental Model Report Card from Ledger history.
#   Prediction: Which invariant did you most consistently underestimate?
#   Reveal: Actual pattern from ledger data (or default if ledger is sparse).
#
# Act II — Full-Stack Design Challenge (20–25 min)
#   Scenario: 7B-parameter LLM inference, P99 < 100 ms, cost < $0.001/req,
#             1000 req/sec, 99.9% availability, all 4 tiers.
#   Student configures: cloud hardware, quantization, serving strategy,
#   tier assignments, and retraining cadence.
#   Live: system architecture diagram, SLO compliance, cost, memory checks.
#   Failure states: P99 violation, cost violation, Edge OOM, TinyML SRAM overflow.
#   Prior ledger entries pre-populate recommendations.
#
# Design Ledger: chapter=16 — full synthesis
# ─────────────────────────────────────────────────────────────────────────────


# ═════════════════════════════════════════════════════════════════════════════
# ZONE A: OPENING
# ═════════════════════════════════════════════════════════════════════════════

# ── CELL 0: SETUP ─────────────────────────────────────────────────────────────
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

    # ── Hardware constants — all plain floats, sources annotated ─────────────

    # Cloud: NVIDIA H100 SXM5
    H100_BW_GBS      = 3350   # GB/s — H100 SXM5 HBM3e, NVIDIA spec
    H100_TFLOPS_FP16 = 989    # TFLOPS FP16 dense tensor core — NVIDIA H100 SXM5 spec
    H100_RAM_GB      = 80     # GB HBM3e on-chip, NVIDIA spec
    H100_TDP_W       = 700    # Watts TDP, NVIDIA spec
    H100_COST_HR     = 3.0    # $/hr cloud on-demand, ~2026 market estimate

    # Edge: NVIDIA Jetson Orin NX 16 GB
    ORIN_BW_GBS      = 102    # GB/s, Jetson Orin NX 16 GB spec
    ORIN_TFLOPS      = 100    # TOPS INT8 equivalent, Jetson Orin NX spec
    ORIN_RAM_GB      = 16     # GB LPDDR5, Jetson Orin NX spec
    ORIN_TDP_W       = 25     # Watts, Jetson Orin NX spec

    # Mobile: Apple A17-class smartphone NPU
    MOBILE_BW_GBS    = 68     # GB/s, Apple A17 measured
    MOBILE_TOPS_INT8 = 35     # TOPS INT8, Apple A17 Neural Engine spec
    MOBILE_RAM_GB    = 8      # GB LPDDR5, typical flagship 2025
    MOBILE_TDP_W     = 5      # Watts sustained thermal budget, measured

    # TinyML: ARM Cortex-M7 (STM32H7-class)
    MCU_BW_GBS       = 0.05   # GB/s — ~400 MB/s AXI bus, Cortex-M7
    MCU_MFLOPS       = 1      # MFLOPS — DSP+FPU, Cortex-M7 at 400 MHz
    MCU_SRAM_KB      = 256    # KB — total on-chip SRAM ceiling, Cortex-M7
    MCU_TDP_MW       = 100    # milliwatts, Cortex-M7 at full compute

    # ── Model: 7B parameter LLM (Llama-3-7B class) ───────────────────────────
    LLM_PARAMS_B     = 7.0    # billion parameters
    LLM_PARAMS       = 7e9    # raw parameter count
    # Memory sizing per quantization:
    #   FP16  = 2 bytes/param  → 7B × 2 = 14 GB
    #   INT8  = 1 byte/param   → 7B × 1 =  7 GB
    #   INT4  = 0.5 bytes/param→ 7B × 0.5 = 3.5 GB
    #   Ternary = ~0.19 bytes  → 7B × 0.19 ≈ 1.3 GB (BitNet-class)
    BYTES_PER_PARAM  = {"fp16": 2.0, "int8": 1.0, "int4": 0.5, "ternary": 0.19}
    # KV cache overhead at batch=1, seq=2048 (rough): ~2 GB FP16 for 7B
    KV_CACHE_GB_FP16 = 2.0    # approximate KV cache at seq_len=2048, batch=1

    # ── SLO targets ──────────────────────────────────────────────────────────
    SLO_P99_MS       = 100    # P99 latency SLO, ms
    SLO_COST_PER_REQ = 0.001  # $/request cost SLO
    SLO_REQ_PER_SEC  = 1000   # required throughput, req/sec

    # ── Accuracy degradation by quantization (empirical baselines) ───────────
    # Source: conclusion.qmd synthesis + ch10 compression lab constants
    ACC_DEGRADATION  = {"fp16": 0.0, "int8": 0.004, "int4": 0.025, "ternary": 0.12}

    # ── Throughput scaling by serving strategy (normalized, H100 baseline) ───
    # Source: model_serving.qmd, Little's Law instrument (Lab 13)
    THROUGHPUT_MULT  = {"static": 0.50, "dynamic": 0.80, "continuous": 1.00}

    ledger = DesignLedger()
    return (
        mo, go, np, math,
        ledger, COLORS, LAB_CSS, apply_plotly_theme,
        H100_BW_GBS, H100_TFLOPS_FP16, H100_RAM_GB, H100_TDP_W, H100_COST_HR,
        ORIN_BW_GBS, ORIN_TFLOPS, ORIN_RAM_GB, ORIN_TDP_W,
        MOBILE_BW_GBS, MOBILE_TOPS_INT8, MOBILE_RAM_GB, MOBILE_TDP_W,
        MCU_BW_GBS, MCU_MFLOPS, MCU_SRAM_KB, MCU_TDP_MW,
        LLM_PARAMS_B, LLM_PARAMS, BYTES_PER_PARAM, KV_CACHE_GB_FP16,
        SLO_P99_MS, SLO_COST_PER_REQ, SLO_REQ_PER_SEC,
        ACC_DEGRADATION, THROUGHPUT_MULT,
    )


# ── CELL 1: HEADER ────────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, LAB_CSS, COLORS):
    _cloud_c  = COLORS["Cloud"]
    _edge_c   = COLORS["Edge"]
    _mobile_c = COLORS["Mobile"]
    _tiny_c   = COLORS["Tiny"]
    mo.vstack([
        LAB_CSS,
        mo.Html(f"""
        <div style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
                    padding: 36px 44px; border-radius: 16px; color: white;
                    box-shadow: 0 8px 32px rgba(0,0,0,0.3);">
            <div style="font-size: 0.72rem; font-weight: 700; letter-spacing: 0.18em;
                        color: #475569; text-transform: uppercase; margin-bottom: 10px;">
                Machine Learning Systems &middot; Volume I &middot; Lab 16 &middot; Capstone
            </div>
            <h1 style="margin: 0 0 10px 0; font-size: 2.4rem; font-weight: 900;
                       color: #f8fafc; line-height: 1.1; letter-spacing: -0.02em;">
                The Architect's Audit
            </h1>
            <p style="margin: 0 0 20px 0; font-size: 1.05rem; color: #94a3b8;
                      max-width: 680px; line-height: 1.65;">
                Sixteen labs. Seven invariants. Four tiers. One capstone challenge.
                Your Design Ledger reveals what your mental model actually looks like —
                and a full-stack 7B LLM deployment forces you to use every lesson at once.
            </p>
            <div style="display: flex; gap: 12px; flex-wrap: wrap;">
                <span style="background: rgba(99,102,241,0.15); color: #a5b4fc;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(99,102,241,0.25);">
                    Act I: Design Ledger Archaeology &middot; 12&ndash;15 min
                </span>
                <span style="background: rgba(99,102,241,0.15); color: #a5b4fc;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(99,102,241,0.25);">
                    Act II: Full-Stack Design Challenge &middot; 20&ndash;25 min
                </span>
                <span style="background: rgba(16,185,129,0.15); color: #6ee7b7;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(16,185,129,0.25);">
                    35&ndash;40 min total
                </span>
                <span style="background: rgba(245,158,11,0.15); color: #fcd34d;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(245,158,11,0.25);">
                    All 4 deployment tiers active
                </span>
            </div>
            <div style="display: flex; gap: 10px; flex-wrap: wrap; margin-top: 14px;">
                <span class="badge badge-info">Iron Law T = D/BW + O/R + L</span>
                <span class="badge badge-info">Amdahl T = t_serial + (1-f)/N</span>
                <span class="badge badge-info">Roofline Ridge Point</span>
                <span class="badge badge-info">Little's Law L = &lambda;W</span>
                <span class="badge badge-warn">All 4 tiers: OOM failure states active</span>
                <span class="badge badge-fail">P99 &lt; 100 ms | cost &lt; $0.001/req</span>
            </div>
        </div>
        """),
    ])
    return


# ── CELL 2: BRIEFING ──────────────────────────────────────────────────────────
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
                <div style="margin-bottom: 3px;">1. <strong>Diagnose your weakest invariant across Labs 01&ndash;15 by reading your Design Ledger prediction accuracy data, not by guessing.</strong></div>
                <div style="margin-bottom: 3px;">2. <strong>Design a full-stack multi-tier deployment that satisfies simultaneous SLAs on P99 latency, cost-per-request, fairness gap, and drift resilience.</strong></div>
                <div style="margin-bottom: 3px;">3. <strong>Identify which of the seven Volume I invariants is the binding constraint for your chosen configuration, and predict how relaxing it by 2&times; changes the system.</strong></div>
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
                    All prior labs (01&ndash;15) assumed complete &middot;
                    Seven Volume I invariants from @sec-conclusion-synthesis &middot;
                    Design philosophy from @sec-conclusion-design-philosophy
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
                "Every sub-team hit its target &mdash; Amdahl ceiling reduced, P99 within SLO, drift detected, fairness audited &mdash; so why did the integrated system lose 4 percentage points of accuracy within weeks, and which of the seven invariants should have predicted this?"
            </div>
        </div>
    </div>
    """)
    return


# ── CELL 3: RECOMMENDED READING ───────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md("""
    **Recommended Reading** — Complete the following before this lab:

    - **@sec-conclusion-synthesis** — The seven invariants of Volume I, their relationships,
      and how they interact in multi-tier deployments.
    - **@sec-conclusion-design-philosophy** — Why constraints drive architecture, and what it
      means to design a system rather than a model.
    - All prior chapter readings for Labs 01–15 are assumed complete.

    *This lab synthesizes every prior lab. No new invariants are introduced.*
    """), kind="info")
    return


# ═════════════════════════════════════════════════════════════════════════════
# ZONE B: ACT I — CALIBRATION
# ═════════════════════════════════════════════════════════════════════════════

# ─── CELL 5: ACT1_BANNER ──────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, COLORS):
    _act_num = "I"
    _act_color = COLORS["BlueLine"]
    _act_title = "Design Ledger Archaeology"
    _act_duration = "12\u201315 min"
    _act_why = (
        "You believe you know your weakest invariant from memory. Your Design Ledger knows it "
        "from data. The prediction gap \u2014 between what you think you got wrong and what the "
        "ledger shows you got wrong \u2014 is itself a systems insight: mental models degrade, "
        "measurement does not."
    )
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


# ── CELL 6: ACT1_STAKEHOLDER ──────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, COLORS):
    _color = COLORS["BlueLine"]
    mo.vstack([
        mo.md("---"),
        mo.md("## Act I — Design Ledger Archaeology"),
        mo.Html(f"""
        <div style="border-left:4px solid {_color}; background:{COLORS['BlueL']};
                    border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
            <div style="font-size:0.72rem; font-weight:700; color:{_color};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                Incoming Message &middot; Chief Architect
            </div>
            <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                "Before I put you on the 7B deployment, I want to see your self-assessment.
                Sixteen labs. Which physical law did your intuition most consistently
                get wrong? Don't guess — look at your data.
                The best engineers know where their blind spots are."
            </div>
        </div>
        """),
        mo.md("""
        Your Design Ledger has recorded every prediction you made and every constraint
        you hit across Labs 01 through 15. The patterns in that data reveal your current
        mental model — specifically, which invariants your intuition underweights.

        Before looking at the ledger summary, commit to your own diagnosis.
        """),
    ])
    return


# ── CELL 4: ACT I PREDICTION LOCK ─────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    act1_prediction = mo.ui.radio(
        options={
            "A)  Memory bandwidth — I underestimated how often bandwidth, not compute, is the bottleneck (Iron Law / Roofline)": "memory_bw",
            "B)  Queuing effects — I underestimated how P99 diverges from average at high utilization (Little's Law / P99)": "queuing",
            "C)  Serial fraction — I overestimated how much parallelism helps (Amdahl's Law / benchmarking)": "serial_fraction",
            "D)  Distribution drift — I underestimated how quickly production data diverges from training (PSI / MLOps)": "drift",
        },
        label="Looking back at Labs 01–15, which invariant did you most consistently underestimate?",
    )
    mo.vstack([
        act1_prediction,
    ])
    return (act1_prediction,)


@app.cell(hide_code=True)
def _(mo, act1_prediction):
    mo.stop(
        act1_prediction.value is None,
        mo.callout(mo.md("Select your prediction above to continue to the Design Ledger summary."), kind="warn"),
    )
    return


# ── CELL 5: LEDGER READER — extract history ───────────────────────────────────
@app.cell(hide_code=True)
def _(ledger):
    # Read entire ledger history. Gracefully handle missing entries.
    # Each entry is {"chapter": N, "design": {...}}
    _history = ledger._state.history

    # Build a lookup: chapter -> design dict (last write wins)
    _ch_data = {}
    for _entry in _history:
        _ch = _entry.get("chapter", 0)
        _d  = _entry.get("design", {})
        _ch_data[_ch] = _d

    # ── Count labs completed ──────────────────────────────────────────────────
    _lab_chapters = list(range(1, 16))  # Labs 01–15
    _completed = [c for c in _lab_chapters if c in _ch_data]
    n_completed = len(_completed)

    # ── Prediction accuracy ───────────────────────────────────────────────────
    _correct_list  = [_ch_data[c].get("act1_correct", False) for c in _completed]
    n_predictions  = len(_correct_list)
    n_correct      = sum(1 for v in _correct_list if v)
    pct_correct    = (n_correct / n_predictions * 100) if n_predictions > 0 else 0.0

    # ── Constraint hits ───────────────────────────────────────────────────────
    _hit_list    = [_ch_data[c].get("constraint_hit", False) for c in _completed]
    n_hits       = sum(1 for v in _hit_list if v)
    hit_chapters = [c for c in _completed if _ch_data[c].get("constraint_hit", False)]

    # ── Context preference ────────────────────────────────────────────────────
    _ctx_counts = {"cloud": 0, "edge": 0, "mobile": 0, "tiny": 0}
    for c in _completed:
        ctx = _ch_data[c].get("context", "cloud")
        if ctx in _ctx_counts:
            _ctx_counts[ctx] += 1
    fav_context = max(_ctx_counts, key=lambda k: _ctx_counts[k]) if n_completed > 0 else "cloud"

    # ── Map chapters to invariant categories ─────────────────────────────────
    # Labels for radar chart (7 categories matching the 7 Volume I invariants)
    INVARIANT_CHAPTERS = {
        "Iron Law":    [2],           # ch02 ml_systems
        "Memory Wall": [5, 8],        # ch05 nn_compute, ch08 training
        "Amdahl":      [6, 12],       # ch06 nn_arch, ch12 benchmarking
        "Roofline":    [7, 11],       # ch07 frameworks, ch11 hw_accel
        "Little's Law":[13],          # ch13 model_serving
        "Drift":       [3, 14],       # ch03 ml_workflow, ch14 ml_ops
        "Fairness":    [15],          # ch15 responsible_engr
    }

    # For each invariant, compute: (correct_rate, constraint_hit_rate) across its chapters
    invariant_scores = {}
    for inv_name, chapters in INVARIANT_CHAPTERS.items():
        _inv_completed = [c for c in chapters if c in _ch_data]
        if not _inv_completed:
            invariant_scores[inv_name] = {"correct_rate": 0.5, "hit_rate": 0.0, "n": 0}
        else:
            _inv_correct = sum(1 for c in _inv_completed if _ch_data[c].get("act1_correct", False))
            _inv_hits    = sum(1 for c in _inv_completed if _ch_data[c].get("constraint_hit", False))
            invariant_scores[inv_name] = {
                "correct_rate": _inv_correct / len(_inv_completed),
                "hit_rate":     _inv_hits    / len(_inv_completed),
                "n":            len(_inv_completed),
            }

    # ── Identify weakest invariant (lowest prediction correct rate) ──────────
    _filled = {k: v for k, v in invariant_scores.items() if v["n"] > 0}
    if _filled:
        weakest_invariant = min(_filled, key=lambda k: _filled[k]["correct_rate"])
    else:
        weakest_invariant = "Memory Wall"   # default if no ledger data

    # ── Map weakest invariant to prediction answer ────────────────────────────
    _inv_to_pred = {
        "Iron Law":    "memory_bw",
        "Memory Wall": "memory_bw",
        "Roofline":    "memory_bw",
        "Little's Law":"queuing",
        "Amdahl":      "serial_fraction",
        "Drift":       "drift",
        "Fairness":    "drift",
    }
    act1_correct_answer = _inv_to_pred.get(weakest_invariant, "memory_bw")

    # ── Prior chapter recommendations for Act II ─────────────────────────────
    # Read specific decisions to pre-populate Act II guidance
    prior_compression = _ch_data.get(10, {}).get("act2_decision", "none")
    prior_p99_ms      = _ch_data.get(13, {}).get("act2_result", 250.0)
    prior_cadence     = _ch_data.get(14, {}).get("act2_decision", "none")

    return (
        _ch_data,
        n_completed, n_predictions, n_correct, pct_correct,
        n_hits, hit_chapters,
        fav_context, _ctx_counts,
        invariant_scores, weakest_invariant, act1_correct_answer,
        prior_compression, prior_p99_ms, prior_cadence,
    )


# ── CELL 6: MENTAL MODEL REPORT CARD ─────────────────────────────────────────
@app.cell(hide_code=True)
def _(
    mo, COLORS,
    n_completed, n_predictions, n_correct, pct_correct,
    n_hits, hit_chapters,
    fav_context, _ctx_counts,
    weakest_invariant,
):
    # Construct the report card display
    _ctx_labels = {"cloud": "Cloud (H100)", "edge": "Edge (Jetson Orin)",
                   "mobile": "Mobile (Smartphone NPU)", "tiny": "TinyML (Cortex-M7)"}
    _fav_display = _ctx_labels.get(fav_context, fav_context)

    _pct_color = (
        COLORS["GreenLine"]  if pct_correct >= 70
        else COLORS["OrangeLine"] if pct_correct >= 45
        else COLORS["RedLine"]
    )
    _hit_color = COLORS["RedLine"] if n_hits > 4 else (
        COLORS["OrangeLine"] if n_hits > 1 else COLORS["GreenLine"]
    )

    _hit_ch_str = ", ".join(f"Lab {c:02d}" for c in hit_chapters) if hit_chapters else "None"

    _ctx_bar = ""
    for _ctx_key, _ctx_count in _ctx_counts.items():
        _ctx_label = _ctx_labels.get(_ctx_key, _ctx_key)
        _ctx_color = COLORS.get(_ctx_key.capitalize(), COLORS["BlueLine"])
        _ctx_pct   = (_ctx_count / max(n_completed, 1)) * 100
        _ctx_bar += f"""
        <div style="display:flex; align-items:center; gap:12px; margin:5px 0;">
            <div style="width:110px; font-size:0.8rem; color:{COLORS['TextSec']};">{_ctx_label}</div>
            <div style="flex:1; background:#f1f5f9; border-radius:6px; height:12px; overflow:hidden;">
                <div style="width:{_ctx_pct:.0f}%; background:{_ctx_color};
                            height:100%; border-radius:6px;"></div>
            </div>
            <div style="width:28px; font-size:0.8rem; font-weight:700;
                        color:{_ctx_color}; text-align:right;">{_ctx_count}</div>
        </div>
        """

    mo.Html(f"""
    <div style="background:white; border:1px solid {COLORS['Border']}; border-radius:14px;
                padding:28px 32px; margin:16px 0; box-shadow:0 2px 12px rgba(0,0,0,0.05);">
        <div style="font-size:0.72rem; font-weight:700; letter-spacing:0.14em;
                    text-transform:uppercase; color:{COLORS['TextMuted']}; margin-bottom:16px;">
            Mental Model Report Card &mdash; Volume I, Labs 01&ndash;15
        </div>

        <div style="display:grid; grid-template-columns:repeat(3,1fr); gap:18px; margin-bottom:24px;">

            <div style="text-align:center; padding:18px; background:{COLORS['BlueLL']};
                        border-radius:10px; border:1px solid {COLORS['BlueL']};">
                <div style="font-size:2.2rem; font-weight:900; color:{COLORS['BlueLine']};">
                    {n_completed}/15
                </div>
                <div style="font-size:0.78rem; color:{COLORS['TextSec']}; margin-top:4px;">
                    Labs Completed
                </div>
            </div>

            <div style="text-align:center; padding:18px; background:{COLORS['GreenLL']};
                        border-radius:10px; border:1px solid {COLORS['GreenL']};">
                <div style="font-size:2.2rem; font-weight:900; color:{_pct_color};">
                    {pct_correct:.0f}%
                </div>
                <div style="font-size:0.78rem; color:{COLORS['TextSec']}; margin-top:4px;">
                    Predictions Correct ({n_correct}/{n_predictions})
                </div>
            </div>

            <div style="text-align:center; padding:18px; background:{COLORS['RedLL']};
                        border-radius:10px; border:1px solid {COLORS['RedL']};">
                <div style="font-size:2.2rem; font-weight:900; color:{_hit_color};">
                    {n_hits}
                </div>
                <div style="font-size:0.78rem; color:{COLORS['TextSec']}; margin-top:4px;">
                    Constraint Failures Triggered
                </div>
            </div>

        </div>

        <div style="display:grid; grid-template-columns:1fr 1fr; gap:24px;">
            <div>
                <div style="font-size:0.8rem; font-weight:700; color:{COLORS['Text']};
                            margin-bottom:10px;">
                    Deployment Context Distribution
                </div>
                {_ctx_bar}
            </div>
            <div>
                <div style="font-size:0.8rem; font-weight:700; color:{COLORS['Text']};
                            margin-bottom:10px;">
                    Diagnosis
                </div>
                <div style="font-size:0.85rem; line-height:1.65; color:{COLORS['TextSec']};">
                    <div style="margin-bottom:6px;">
                        <strong>Weakest invariant:</strong>
                        <span style="color:{COLORS['OrangeLine']}; font-weight:700;">
                            {weakest_invariant}
                        </span>
                    </div>
                    <div style="margin-bottom:6px;">
                        <strong>Constraint failures at:</strong>
                        <span style="color:{COLORS['RedLine']};">{_hit_ch_str}</span>
                    </div>
                    <div>
                        <strong>Favorite context:</strong>
                        <span style="color:{COLORS['BlueLine']};">{_fav_display}</span>
                    </div>
                </div>
            </div>
        </div>

        {"<div style='margin-top:16px; padding:12px 16px; background:#fffbeb; border-radius:8px; border:1px solid #fde68a; font-size:0.82rem; color:#78350f;'><strong>Note:</strong> If fewer than 15 labs are recorded, some metrics default to neutral estimates. Complete missing labs to see your full report.</div>" if n_completed < 15 else ""}
    </div>
    """)
    return


# ── CELL 7: RADAR CHART — invariant performance ───────────────────────────────
@app.cell(hide_code=True)
def _(mo, go, COLORS, invariant_scores, apply_plotly_theme):
    _cats = list(invariant_scores.keys())
    _correct_rates = [invariant_scores[c]["correct_rate"] * 100 for c in _cats]
    _hit_rates     = [invariant_scores[c]["hit_rate"]     * 100 for c in _cats]

    # Close the polygon
    _cats_closed   = _cats + [_cats[0]]
    _correct_closed = _correct_rates + [_correct_rates[0]]
    _hit_closed     = _hit_rates     + [_hit_rates[0]]

    _fig = go.Figure()

    _fig.add_trace(go.Scatterpolar(
        r=_correct_closed,
        theta=_cats_closed,
        fill="toself",
        name="Prediction Accuracy (%)",
        line=dict(color=COLORS["GreenLine"], width=2),
        fillcolor=f"rgba(0,143,69,0.12)",
    ))
    _fig.add_trace(go.Scatterpolar(
        r=_hit_closed,
        theta=_cats_closed,
        fill="toself",
        name="Constraint Hit Rate (%)",
        line=dict(color=COLORS["RedLine"], width=2, dash="dot"),
        fillcolor=f"rgba(203,32,45,0.10)",
    ))

    _fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                gridcolor="#f1f5f9",
                tickcolor=COLORS["TextMuted"],
                tickfont=dict(size=9),
            ),
            angularaxis=dict(
                tickfont=dict(size=11, color=COLORS["Text"]),
                gridcolor="#e2e8f0",
            ),
            bgcolor="white",
        ),
        showlegend=True,
        legend=dict(orientation="h", y=-0.12, font=dict(size=10)),
        height=380,
        margin=dict(l=60, r=60, t=30, b=40),
        paper_bgcolor="white",
        font=dict(family="Inter, sans-serif"),
        title=dict(
            text="Invariant Performance Map — Prediction Accuracy vs Constraint Hit Rate",
            font=dict(size=12, color=COLORS["TextSec"]),
            x=0.5,
        ),
    )

    mo.vstack([
        mo.md("### Performance Across the 7 Invariants"),
        mo.md("""
        Green area: how often your prediction was correct for each invariant category.
        Red dashed area: how often you triggered the failure state (constraint hit).
        A high constraint-hit rate means the physics surprised you — your mental model
        underestimated that invariant's real-world severity.
        """),
        mo.ui.plotly(_fig),
    ])
    return


# ── CELL 8: ACT I PREDICTION REVEAL ──────────────────────────────────────────
@app.cell(hide_code=True)
def _(
    mo, COLORS,
    act1_prediction, act1_correct_answer, weakest_invariant,
    pct_correct, n_hits, n_completed,
):
    _chosen  = act1_prediction.value
    _correct = (_chosen == act1_correct_answer)

    _inv_explanations = {
        "memory_bw": (
            "**Memory bandwidth** (Iron Law / Roofline) is the most common blind spot "
            "for engineers trained primarily on algorithmic complexity. "
            "FLOPs double every two years on new hardware, but memory bandwidth grows "
            "much more slowly. A model that is *compute-bound* on paper is often "
            "*memory-bound* in practice because weights must transit from HBM to SM "
            "on every forward pass. The Iron Law `T = D/BW + O/R + L` makes this "
            "concrete: when D/BW dominates, doubling compute does nothing."
        ),
        "queuing": (
            "**Queuing effects** (Little's Law / P99) are invisible until they are catastrophic. "
            "Average latency is a *mean* — it hides the tail. At 80% utilization, "
            "M/M/1 theory predicts P99 explodes by 5–10× relative to average. "
            "The engineering trap is optimizing the dashboard metric (average) while "
            "the SLO metric (P99) silently violates. Little's Law `L = λW` quantifies "
            "exactly when the queue becomes the bottleneck, not the model."
        ),
        "serial_fraction": (
            "**Serial fraction** (Amdahl's Law) is consistently underestimated because "
            "engineers conflate 'adding GPUs' with 'getting proportional speedup.' "
            "A 5% serial fraction caps speedup at 20× regardless of how many GPUs "
            "you add. In practice, data loading, synchronization barriers, and "
            "preprocessing pipelines introduce serial fractions that make Amdahl's "
            "ceiling hit much sooner than expected."
        ),
        "drift": (
            "**Distribution drift** (PSI / MLOps) is counterintuitive because the model "
            "code doesn't change — the world does. Engineers trained on static benchmarks "
            "have no intuitive feel for how quickly production data diverges from "
            "training data. Population Stability Index (PSI) provides a quantitative "
            "signal, but without monitoring infrastructure, drift is silent. "
            "The Lab 03 silent degradation scenario is the canonical example."
        ),
    }

    _explanation = _inv_explanations.get(_chosen or "memory_bw", "")
    _kind = "success" if _correct else "warn"

    _ledger_diagnosis = (
        f"Based on your Design Ledger ({n_completed} labs completed), "
        f"your weakest invariant by prediction accuracy was **{weakest_invariant}**. "
        f"You triggered constraint failures in {n_hits} labs, "
        f"and predicted correctly {pct_correct:.0f}% of the time overall."
        if n_completed > 0
        else
        "Your Design Ledger is sparse (fewer labs completed than expected). "
        "The analysis defaults to the most statistically common blind spot "
        "across the student population: **Memory Wall / Roofline**. "
        "Complete prior labs to generate your personalized report."
    )

    mo.vstack([
        mo.callout(
            mo.md(
                f"**{'Confirmed.' if _correct else "Different reading."}** {_explanation}\n\n"
                f"*Ledger reading:* {_ledger_diagnosis}"
            ),
            kind=_kind,
        ),
    ])
    return


# ── CELL 9: ACT I MATHPEEK ────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "The 7 Invariants of Volume I — Governing Equations": mo.md("""
        These are the physical laws your system designs must satisfy.
        Each appears as an active constraint in Act II.

        **1. Iron Law (Memory Bandwidth)**
        ```
        T_total = D / BW + O / R + L
        ```
        D = data bytes accessed, BW = memory bandwidth, O = operations,
        R = throughput (ops/sec), L = latency overhead.

        **2. Memory Wall (Model Footprint)**
        ```
        Memory_GB = Params × bytes_per_param + KV_cache + activations
        ```
        For FP16: 2 bytes/param. For INT8: 1 byte/param. For INT4: 0.5 bytes/param.

        **3. Amdahl's Law (Parallelism Ceiling)**
        ```
        Speedup = 1 / (S + (1-S)/N)
        ```
        S = serial fraction, N = number of parallel units.
        Maximum speedup = 1/S regardless of N.

        **4. Roofline Model (Performance Ceiling)**
        ```
        Performance = min(Peak_FLOPS, Arithmetic_Intensity × Memory_BW)
        ```
        Ridge point: AI* = Peak_FLOPS / Memory_BW. Below AI*: memory-bound.

        **5. Little's Law (Queue Depth)**
        ```
        L = λ × W
        ```
        L = mean queue depth, λ = arrival rate, W = mean time in system.
        At utilization ρ → 1, W → ∞ (M/M/1 queue diverges).

        **6. Degradation Equation (Model Drift)**
        ```
        Accuracy(t) = Accuracy_0 − λ_sensitivity × PSI(t)
        ```
        PSI = Population Stability Index. λ_sensitivity = sensitivity coefficient.

        **7. Fairness-Accuracy Tradeoff (Chouldechova 2017)**
        ```
        When base_rate_A ≠ base_rate_B:
        DP ∩ EO ∩ CA = ∅   (no classifier can satisfy all three)
        ```
        DP = demographic parity, EO = equalized odds, CA = calibration.
        """),
    })
    return


# ─────────────────────────────────────────────────────────────────────────────
# ═════════════════════════════════════════════════════════════════════════════
# ZONE C: ACT II — DESIGN CHALLENGE
# ═════════════════════════════════════════════════════════════════════════════

# ─── CELL 12: ACT2_BANNER ─────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, COLORS):
    _act_num = "II"
    _act_color = COLORS["OrangeLine"]
    _act_title = "Full-Stack Design Challenge"
    _act_duration = "20\u201325 min"
    _act_why = (
        "Act I revealed your weakest invariant. Now all seven activate simultaneously: "
        "the system you configure must satisfy P99, cost, memory, fairness, and drift "
        "constraints at once \u2014 and optimizing any one will shift cost to another, "
        "because complexity in an ML system cannot be destroyed, only moved."
    )
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


# ── CELL 13: ACT II INTRO ─────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(
    mo, COLORS,
    prior_compression, prior_p99_ms, prior_cadence,
    SLO_P99_MS, SLO_COST_PER_REQ, SLO_REQ_PER_SEC,
    LLM_PARAMS_B,
):
    _color = COLORS["Cloud"]
    # Build contextual recommendations from prior labs
    _recs = []
    if prior_compression not in ("none", ""):
        _recs.append(f"Your Lab 10 compression choice was **{prior_compression}** — consider this for mobile and edge tiers.")
    if isinstance(prior_p99_ms, (int, float)) and prior_p99_ms > 200:
        _recs.append(f"Your Lab 13 P99 was **{prior_p99_ms:.0f} ms** at your chosen load — the 100 ms SLO here is significantly tighter.")
    if prior_cadence not in ("none", ""):
        _recs.append(f"Your Lab 14 retraining choice was **{prior_cadence}** — the synthesis challenge uses all 4 tiers, which have different drift rates.")

    _recs_html = ""
    if _recs:
        _rec_color = COLORS["TextSec"]
        _rec_items = "".join(
            f"<div style='font-size:0.83rem; color:{_rec_color}; margin-bottom:4px;'>&bull; {r}</div>"
            for r in _recs
        )
        _recs_html = f"""
        <div style="margin-top:14px; padding:12px 16px; background:{COLORS['BlueLL']};
                    border-radius:8px; border:1px solid {COLORS['BlueL']};">
            <div style="font-size:0.75rem; font-weight:700; color:{COLORS['BlueLine']};
                        text-transform:uppercase; letter-spacing:0.08em; margin-bottom:8px;">
                Recommendations from your prior labs
            </div>
            {_rec_items}
        </div>
        """

    mo.vstack([
        mo.md("---"),
        mo.md("## Act II — Full-Stack Design Challenge"),
        mo.Html(f"""
        <div style="border-left:4px solid {_color}; background:{COLORS['BlueL']};
                    border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
            <div style="font-size:0.72rem; font-weight:700; color:{_color};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                System Brief &middot; Lead Architect
            </div>
            <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                "You are the lead architect for a new {LLM_PARAMS_B:.0f}B-parameter LLM inference
                system. Requirements: P99 &lt; {SLO_P99_MS} ms, cost &lt; ${SLO_COST_PER_REQ}/req,
                {SLO_REQ_PER_SEC:,} req/sec aggregate, 99.9% availability.
                Deployment spans all four tiers: cloud anchor, edge cache,
                mobile local, and TinyML trigger. Every design decision must satisfy
                the physical constraints — no exceptions, no hand-waving."
            </div>
            {_recs_html}
        </div>
        """),
        mo.md("""
        Work through the five design stages below. Each stage activates a constraint
        from the invariants you have studied. The system architecture diagram updates live.
        Trigger at least one failure state — then fix it.
        """),
    ])
    return


# ── CELL 11: ACT II PREDICTION LOCK ──────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    act2_prediction = mo.ui.radio(
        options={
            "A)  The cloud tier will be the primary bottleneck — model doesn't fit in H100 memory at FP16": "cloud_oom",
            "B)  The edge tier will be the primary bottleneck — Jetson Orin can't serve P99 < 100 ms at scale": "edge_latency",
            "C)  The mobile tier will be the primary bottleneck — 7B doesn't fit in 8 GB at any quantization": "mobile_oom",
            "D)  The TinyML tier will be the primary bottleneck — even ternary quantization won't fit in 256 KB SRAM": "tiny_oom",
        },
        label="Before configuring the system: which tier do you predict will be the hardest constraint to satisfy?",
    )
    mo.vstack([
        act2_prediction,
    ])
    return (act2_prediction,)


@app.cell(hide_code=True)
def _(mo, act2_prediction):
    mo.stop(
        act2_prediction.value is None,
        mo.callout(mo.md("Select your prediction above to unlock the design instruments."), kind="warn"),
    )
    return


# ── CELL 12: STAGE 1 — CLOUD HARDWARE ─────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("### Stage 1 — Cloud Hardware")
    return


@app.cell(hide_code=True)
def _(mo):
    n_h100 = mo.ui.slider(
        start=1, stop=8, value=2, step=1,
        label="Number of H100 GPUs (cloud tier)",
    )
    n_h100
    return (n_h100,)


# ── CELL 13: STAGE 2 — QUANTIZATION STRATEGY ─────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("### Stage 2 — Quantization Strategy")
    return


@app.cell(hide_code=True)
def _(mo):
    quant_cloud = mo.ui.dropdown(
        options={
            "FP16 — full precision, highest accuracy":  "fp16",
            "INT8 — 2× compression, ~0.4% accuracy drop": "int8",
            "INT4 — 4× compression, ~2.5% accuracy drop": "int4",
        },
        value="FP16 — full precision, highest accuracy",
        label="Cloud quantization",
    )
    quant_edge = mo.ui.dropdown(
        options={
            "FP16 — full precision":                     "fp16",
            "INT8 — 2× compression, ~0.4% accuracy drop": "int8",
            "INT4 — 4× compression, ~2.5% accuracy drop": "int4",
        },
        value="INT8 — 2× compression, ~0.4% accuracy drop",
        label="Edge (Orin) quantization",
    )
    quant_mobile = mo.ui.dropdown(
        options={
            "INT8 — 1 byte/param":                       "int8",
            "INT4 — 0.5 bytes/param":                    "int4",
            "Ternary — ~0.19 bytes/param, ~12% accuracy drop": "ternary",
        },
        value="INT4 — 0.5 bytes/param",
        label="Mobile (NPU) quantization",
    )
    mo.hstack([quant_cloud, quant_edge, quant_mobile], justify="start", gap="2rem")
    return (quant_cloud, quant_edge, quant_mobile)


# ── CELL 14: STAGE 3 — SERVING STRATEGY ──────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("### Stage 3 — Serving Strategy")
    return


@app.cell(hide_code=True)
def _(mo):
    serving_strategy = mo.ui.radio(
        options={
            "Static batch — simplest, lowest throughput (50% of peak)":      "static",
            "Dynamic batch — moderate complexity, 80% of peak":               "dynamic",
            "Continuous batching — highest throughput (100% of peak), LLM-optimized": "continuous",
        },
        value="Dynamic batch — moderate complexity, 80% of peak",
        label="Cloud serving strategy (all tiers use dynamic by default):",
        inline=False,
    )
    serving_strategy
    return (serving_strategy,)


# ── CELL 15: STAGE 4 — RETRAINING CADENCE ────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("### Stage 4 — Retraining Cadence")
    return


@app.cell(hide_code=True)
def _(mo):
    retrain_cadence_days = mo.ui.slider(
        start=1, stop=90, value=30, step=1,
        label="Retraining cadence (days between full retraining runs)",
    )
    retrain_cadence_days
    return (retrain_cadence_days,)


# ── CELL 16: STAGE 5 — FAIRNESS THRESHOLD ────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("### Stage 5 — Fairness Monitoring Threshold")
    return


@app.cell(hide_code=True)
def _(mo):
    fairness_threshold_pct = mo.ui.slider(
        start=0, stop=20, value=10, step=1,
        label="Maximum allowed approval gap between demographic groups (%)",
    )
    fairness_threshold_pct
    return (fairness_threshold_pct,)


# ── CELL 17: PHYSICS ENGINE — compute all metrics ─────────────────────────────
@app.cell(hide_code=True)
def _(
    mo,
    n_h100, quant_cloud, quant_edge, quant_mobile, serving_strategy,
    retrain_cadence_days, fairness_threshold_pct,
    H100_BW_GBS, H100_TFLOPS_FP16, H100_RAM_GB, H100_TDP_W, H100_COST_HR,
    ORIN_BW_GBS, ORIN_RAM_GB,
    MOBILE_RAM_GB,
    MCU_BW_GBS, MCU_SRAM_KB,
    LLM_PARAMS, BYTES_PER_PARAM, KV_CACHE_GB_FP16,
    SLO_P99_MS, SLO_COST_PER_REQ, SLO_REQ_PER_SEC,
    ACC_DEGRADATION, THROUGHPUT_MULT,
    COLORS, apply_plotly_theme, go,
):
    # ── Unpack widget values ───────────────────────────────────────────────────
    _n_gpus   = n_h100.value
    _qc       = quant_cloud.value
    _qe       = quant_edge.value
    _qm       = quant_mobile.value
    _serving  = serving_strategy.value
    _cadence  = retrain_cadence_days.value
    _gap_thr  = fairness_threshold_pct.value

    # ── Memory footprints (GB) ─────────────────────────────────────────────────
    # KV cache scales with quantization precision (approx)
    _kv_scale = {"fp16": 1.0, "int8": 0.5, "int4": 0.25, "ternary": 0.09}
    cloud_model_gb  = LLM_PARAMS * BYTES_PER_PARAM[_qc] / 1e9 + KV_CACHE_GB_FP16 * _kv_scale[_qc]
    edge_model_gb   = LLM_PARAMS * BYTES_PER_PARAM[_qe] / 1e9 + KV_CACHE_GB_FP16 * _kv_scale[_qe]
    mobile_model_gb = LLM_PARAMS * BYTES_PER_PARAM[_qm] / 1e9 + KV_CACHE_GB_FP16 * _kv_scale[_qm]
    # TinyML: must use ternary quantization + aggressive pruning
    # Assume pruning reduces to 10M params for a keyword/trigger sub-model
    MCU_EFFECTIVE_PARAMS = 10e6   # 10M param sub-model for TinyML trigger
    tiny_model_kb = MCU_EFFECTIVE_PARAMS * BYTES_PER_PARAM["ternary"] * 1024 / 1e6  # KB

    # ── Feasibility checks ─────────────────────────────────────────────────────
    # Cloud: model must fit across N GPUs (tensor-parallel sharding)
    cloud_ram_available = _n_gpus * H100_RAM_GB
    cloud_oom  = cloud_model_gb > cloud_ram_available

    # Edge: Jetson Orin NX 16 GB
    edge_oom   = edge_model_gb > ORIN_RAM_GB

    # Mobile: smartphone NPU 8 GB
    mobile_oom = mobile_model_gb > MOBILE_RAM_GB

    # TinyML: 256 KB SRAM hard ceiling
    tiny_oom   = tiny_model_kb > MCU_SRAM_KB

    # ── Throughput / latency (Iron Law approximation) ──────────────────────────
    # Cloud: tokens/sec capacity = BW / bytes_per_token (memory-bound at decode)
    # For a 7B INT8 model, each token generation requires reading all weights once
    # Source: Iron Law from ml_systems.qmd, Lab 02 formula
    _bpw_cloud = BYTES_PER_PARAM[_qc]
    _bpw_edge  = BYTES_PER_PARAM[_qe]

    cloud_tok_per_sec  = (H100_BW_GBS * 1e9 / (LLM_PARAMS * _bpw_cloud)) * _n_gpus
    edge_tok_per_sec   = ORIN_BW_GBS  * 1e9 / (LLM_PARAMS * _bpw_edge)

    # Assume output_len = 256 tokens; latency = output_len / tok_per_sec
    OUTPUT_TOKENS = 256
    _serving_mult = THROUGHPUT_MULT[_serving]

    cloud_p99_ms  = (OUTPUT_TOKENS / (cloud_tok_per_sec  * _serving_mult)) * 1000
    edge_p99_ms   = (OUTPUT_TOKENS / (edge_tok_per_sec   * 0.8)           ) * 1000
    # Mobile: approximation using TOPS-based throughput (INT8/INT4)
    mobile_p99_ms = (OUTPUT_TOKENS / max((MOBILE_RAM_GB / mobile_model_gb) * 20, 0.01)) * 1000

    # ── Aggregate throughput (all tiers combined) ──────────────────────────────
    # Assume traffic split: 60% cloud, 25% edge, 14% mobile, 1% tiny
    cloud_rps  = SLO_REQ_PER_SEC * 0.60
    edge_rps   = SLO_REQ_PER_SEC * 0.25
    mobile_rps = SLO_REQ_PER_SEC * 0.14

    cloud_can_handle  = cloud_tok_per_sec  * _serving_mult >= cloud_rps  * OUTPUT_TOKENS
    edge_can_handle   = edge_tok_per_sec   * 0.8           >= edge_rps   * OUTPUT_TOKENS
    mobile_can_handle = not mobile_oom  # if it fits, mobile NPU handles its fraction

    # ── Cost model (cloud compute dominant) ───────────────────────────────────
    # H100 runs 24/7 to handle 1000 req/sec
    # Cost/req = (N_GPUs * $/hr) / (req/hr capacity)
    _req_per_hr = cloud_rps * 3600
    cost_per_req_usd = (_n_gpus * H100_COST_HR) / max(_req_per_hr, 1)

    # ── Accuracy after quantization ───────────────────────────────────────────
    # Worst-case accuracy loss across tiers (mobile is most aggressive)
    max_acc_degradation = max(
        ACC_DEGRADATION[_qc],
        ACC_DEGRADATION[_qe],
        ACC_DEGRADATION[_qm],
    )
    effective_accuracy = 1.0 - max_acc_degradation

    # ── Drift risk from retraining cadence ────────────────────────────────────
    # From degradation equation (Lab 14 / ml_ops.qmd):
    # Accuracy(t) = Acc_0 − λ × PSI(t); PSI grows roughly linearly with days
    # λ_sensitivity ≈ 0.001/day (empirical from chapter scenario)
    LAMBDA_DRIFT = 0.001
    drift_accuracy_loss = LAMBDA_DRIFT * _cadence
    post_drift_accuracy = effective_accuracy - drift_accuracy_loss

    # ── SLO compliance ────────────────────────────────────────────────────────
    slo_latency_ok  = cloud_p99_ms <= SLO_P99_MS
    slo_cost_ok     = cost_per_req_usd <= SLO_COST_PER_REQ
    slo_tput_ok     = cloud_can_handle

    # ── Overall system validity ────────────────────────────────────────────────
    system_valid = (
        not cloud_oom and not edge_oom and not mobile_oom and not tiny_oom
        and slo_latency_ok and slo_cost_ok and slo_tput_ok
    )
    constraint_hit = cloud_oom or edge_oom or mobile_oom or tiny_oom or not slo_latency_ok or not slo_cost_ok

    # ── Tier assignments dict ─────────────────────────────────────────────────
    tier_assignments = {
        "cloud":  _qc,
        "edge":   _qe,
        "mobile": _qm,
        "tiny":   "ternary (sub-model)",
    }

    # ── Status colors for display ─────────────────────────────────────────────
    def _ok_color(ok):
        return COLORS["GreenLine"] if ok else COLORS["RedLine"]

    def _ok_bg(ok):
        return COLORS["GreenLL"] if ok else COLORS["RedLL"]

    def _ok_border(ok):
        return COLORS["GreenL"] if ok else COLORS["RedL"]

    def _ok_label(ok):
        return "OK" if ok else "FAIL"

    # ── Render system metrics panel ───────────────────────────────────────────
    _panel = mo.Html(f"""
    <div style="background:white; border:1px solid {COLORS['Border']}; border-radius:14px;
                padding:24px 28px; margin:16px 0; box-shadow:0 2px 12px rgba(0,0,0,0.05);">
        <div style="font-size:0.72rem; font-weight:700; letter-spacing:0.14em;
                    text-transform:uppercase; color:{COLORS['TextMuted']}; margin-bottom:18px;">
            Live System Architecture — Physics Check
        </div>

        <!-- Physics formulas active -->
        <div style="background:{COLORS['BlueLL']}; border:1px solid {COLORS['BlueL']};
                    border-radius:8px; padding:12px 16px; margin-bottom:18px;
                    font-family:'SF Mono',monospace; font-size:0.78rem; color:{COLORS['BlueLine']};">
            <strong>Iron Law</strong> &nbsp; T_cloud = {OUTPUT_TOKENS} tokens &divide; ({cloud_tok_per_sec * _serving_mult:.1f} tok/s) = <strong>{cloud_p99_ms:.1f} ms</strong><br/>
            <strong>Memory Wall</strong> &nbsp; cloud_mem = {LLM_PARAMS/1e9:.1f}B &times; {BYTES_PER_PARAM[_qc]:.2f} B/param + KV = <strong>{cloud_model_gb:.1f} GB</strong><br/>
            <strong>Cost Model</strong> &nbsp; cost/req = ({_n_gpus} &times; ${H100_COST_HR}/hr) &divide; ({cloud_rps * 3600:,.0f} req/hr) = <strong>${cost_per_req_usd:.5f}</strong>
        </div>

        <!-- 4-tier grid -->
        <div style="display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin-bottom:20px;">

            <!-- Cloud tier -->
            <div style="border-radius:10px; padding:16px; border:2px solid {_ok_border(not cloud_oom)};
                        background:{_ok_bg(not cloud_oom)};">
                <div style="font-size:0.72rem; font-weight:700; color:{COLORS['Cloud']};
                            text-transform:uppercase; margin-bottom:8px;">
                    Cloud (H100 &times;{_n_gpus})
                </div>
                <div style="font-size:0.83rem; line-height:1.65;">
                    <div>Model: <strong>{cloud_model_gb:.1f} GB</strong></div>
                    <div>RAM: <strong>{cloud_ram_available} GB</strong></div>
                    <div>P99: <strong style="color:{_ok_color(slo_latency_ok)};">{cloud_p99_ms:.1f} ms</strong></div>
                    <div>Quant: <strong>{_qc.upper()}</strong></div>
                    <div style="margin-top:8px; font-weight:800; color:{_ok_color(not cloud_oom)};">
                        MEM: {_ok_label(not cloud_oom)}
                    </div>
                </div>
            </div>

            <!-- Edge tier -->
            <div style="border-radius:10px; padding:16px; border:2px solid {_ok_border(not edge_oom)};
                        background:{_ok_bg(not edge_oom)};">
                <div style="font-size:0.72rem; font-weight:700; color:{COLORS['Edge']};
                            text-transform:uppercase; margin-bottom:8px;">
                    Edge (Orin NX)
                </div>
                <div style="font-size:0.83rem; line-height:1.65;">
                    <div>Model: <strong>{edge_model_gb:.1f} GB</strong></div>
                    <div>RAM: <strong>{ORIN_RAM_GB} GB</strong></div>
                    <div>P99: <strong>{edge_p99_ms:.1f} ms</strong></div>
                    <div>Quant: <strong>{_qe.upper()}</strong></div>
                    <div style="margin-top:8px; font-weight:800; color:{_ok_color(not edge_oom)};">
                        MEM: {_ok_label(not edge_oom)}
                    </div>
                </div>
            </div>

            <!-- Mobile tier -->
            <div style="border-radius:10px; padding:16px; border:2px solid {_ok_border(not mobile_oom)};
                        background:{_ok_bg(not mobile_oom)};">
                <div style="font-size:0.72rem; font-weight:700; color:{COLORS['Mobile']};
                            text-transform:uppercase; margin-bottom:8px;">
                    Mobile (NPU)
                </div>
                <div style="font-size:0.83rem; line-height:1.65;">
                    <div>Model: <strong>{mobile_model_gb:.2f} GB</strong></div>
                    <div>RAM: <strong>{MOBILE_RAM_GB} GB</strong></div>
                    <div>P99: <strong>{mobile_p99_ms:.1f} ms</strong></div>
                    <div>Quant: <strong>{_qm.upper()}</strong></div>
                    <div style="margin-top:8px; font-weight:800; color:{_ok_color(not mobile_oom)};">
                        MEM: {_ok_label(not mobile_oom)}
                    </div>
                </div>
            </div>

            <!-- TinyML tier -->
            <div style="border-radius:10px; padding:16px; border:2px solid {_ok_border(not tiny_oom)};
                        background:{_ok_bg(not tiny_oom)};">
                <div style="font-size:0.72rem; font-weight:700; color:{COLORS['Tiny']};
                            text-transform:uppercase; margin-bottom:8px;">
                    TinyML (Cortex-M7)
                </div>
                <div style="font-size:0.83rem; line-height:1.65;">
                    <div>Sub-model: <strong>10M params</strong></div>
                    <div>Footprint: <strong>{tiny_model_kb:.1f} KB</strong></div>
                    <div>SRAM: <strong>{MCU_SRAM_KB} KB</strong></div>
                    <div>Quant: <strong>TERNARY</strong></div>
                    <div style="margin-top:8px; font-weight:800; color:{_ok_color(not tiny_oom)};">
                        MEM: {_ok_label(not tiny_oom)}
                    </div>
                </div>
            </div>

        </div>

        <!-- SLO compliance bar -->
        <div style="display:grid; grid-template-columns:repeat(4,1fr); gap:12px;">
            <div style="text-align:center; padding:12px; border-radius:8px;
                        background:{_ok_bg(slo_latency_ok)}; border:1px solid {_ok_border(slo_latency_ok)};">
                <div style="font-size:0.72rem; font-weight:700; color:{COLORS['TextMuted']};">P99 Latency</div>
                <div style="font-size:1.4rem; font-weight:900; color:{_ok_color(slo_latency_ok)};">
                    {cloud_p99_ms:.1f} ms
                </div>
                <div style="font-size:0.72rem; color:{COLORS['TextMuted']};">SLO: &lt; 100 ms</div>
            </div>
            <div style="text-align:center; padding:12px; border-radius:8px;
                        background:{_ok_bg(slo_cost_ok)}; border:1px solid {_ok_border(slo_cost_ok)};">
                <div style="font-size:0.72rem; font-weight:700; color:{COLORS['TextMuted']};">Cost / Request</div>
                <div style="font-size:1.4rem; font-weight:900; color:{_ok_color(slo_cost_ok)};">
                    ${cost_per_req_usd:.5f}
                </div>
                <div style="font-size:0.72rem; color:{COLORS['TextMuted']};">SLO: &lt; $0.001</div>
            </div>
            <div style="text-align:center; padding:12px; border-radius:8px;
                        background:{_ok_bg(effective_accuracy > 0.90)}; border:1px solid {_ok_border(effective_accuracy > 0.90)};">
                <div style="font-size:0.72rem; font-weight:700; color:{COLORS['TextMuted']};">Accuracy</div>
                <div style="font-size:1.4rem; font-weight:900; color:{_ok_color(effective_accuracy > 0.90)};">
                    {effective_accuracy * 100:.1f}%
                </div>
                <div style="font-size:0.72rem; color:{COLORS['TextMuted']};">Target: &gt; 90%</div>
            </div>
            <div style="text-align:center; padding:12px; border-radius:8px;
                        background:{_ok_bg(post_drift_accuracy > 0.88)}; border:1px solid {_ok_border(post_drift_accuracy > 0.88)};">
                <div style="font-size:0.72rem; font-weight:700; color:{COLORS['TextMuted']};">Post-Drift Accuracy</div>
                <div style="font-size:1.4rem; font-weight:900; color:{_ok_color(post_drift_accuracy > 0.88)};">
                    {post_drift_accuracy * 100:.1f}%
                </div>
                <div style="font-size:0.72rem; color:{COLORS['TextMuted']};">At cadence={_cadence}d</div>
            </div>
        </div>
    </div>
    """)

    return (
        _panel,
        cloud_model_gb, edge_model_gb, mobile_model_gb, tiny_model_kb,
        cloud_oom, edge_oom, mobile_oom, tiny_oom,
        cloud_p99_ms, edge_p99_ms, mobile_p99_ms,
        cost_per_req_usd, effective_accuracy, post_drift_accuracy,
        slo_latency_ok, slo_cost_ok, slo_tput_ok,
        system_valid, constraint_hit,
        tier_assignments, _n_gpus, _qc, _qe, _qm, _serving, _cadence,
    )


# ── CELL 18: RENDER METRICS PANEL ─────────────────────────────────────────────
@app.cell(hide_code=True)
def _(_panel):
    _panel
    return


# ── CELL 19: FAILURE STATE BANNERS ────────────────────────────────────────────
@app.cell(hide_code=True)
def _(
    mo,
    cloud_oom, edge_oom, mobile_oom, tiny_oom,
    slo_latency_ok, slo_cost_ok,
    cloud_model_gb, cloud_p99_ms, cost_per_req_usd,
    edge_model_gb, mobile_model_gb, tiny_model_kb,
    H100_RAM_GB, n_h100, ORIN_RAM_GB, MOBILE_RAM_GB, MCU_SRAM_KB,
    SLO_P99_MS, SLO_COST_PER_REQ,
    effective_accuracy, post_drift_accuracy,
    retrain_cadence_days,
):
    _banners = []
    _n = n_h100.value
    _cadence = retrain_cadence_days.value

    if cloud_oom:
        _banners.append(mo.callout(mo.md(
            f"**OOM — Cloud Tier Infeasible.** "
            f"Required: **{cloud_model_gb:.1f} GB** | Available: **{_n * H100_RAM_GB} GB** "
            f"({_n} × H100 @ 80 GB each). "
            f"Fix: increase GPU count to {int(cloud_model_gb / H100_RAM_GB) + 1}+, "
            f"or switch to INT8 quantization (7 GB model)."
        ), kind="danger"))

    if edge_oom:
        _banners.append(mo.callout(mo.md(
            f"**OOM — Edge Tier Infeasible.** "
            f"Required: **{edge_model_gb:.1f} GB** | Jetson Orin NX available: **{ORIN_RAM_GB} GB**. "
            f"The 7B model at FP16 (14 GB) exceeds Orin's 16 GB only with overhead included. "
            f"Fix: quantize edge tier to INT8 (≈7 GB) or INT4 (≈3.5 GB)."
        ), kind="danger"))

    if mobile_oom:
        _banners.append(mo.callout(mo.md(
            f"**OOM — Mobile Tier Infeasible.** "
            f"Required: **{mobile_model_gb:.2f} GB** | Smartphone NPU available: **{MOBILE_RAM_GB} GB**. "
            f"The 7B model at INT8 (7 GB) just fits; FP16 (14 GB) and INT4+KV overhead "
            f"can also exceed budget. Fix: switch mobile tier to INT4 (≈3.5 GB) or Ternary."
        ), kind="danger"))

    if tiny_oom:
        _banners.append(mo.callout(mo.md(
            f"**SRAM Overflow — TinyML Tier Infeasible.** "
            f"Required: **{tiny_model_kb:.1f} KB** | Cortex-M7 SRAM ceiling: **{MCU_SRAM_KB} KB**. "
            f"The 10M-param ternary sub-model footprint is {tiny_model_kb:.1f} KB. "
            f"If this exceeds 256 KB, further pruning is required — target &lt; 5M params "
            f"or replace with a 100K-param keyword-spotting model."
        ), kind="danger"))

    if not slo_latency_ok:
        _banners.append(mo.callout(mo.md(
            f"**P99 SLO Violated.** "
            f"Cloud P99: **{cloud_p99_ms:.1f} ms** exceeds the **{SLO_P99_MS} ms** SLO. "
            f"The Iron Law `T = D/BW + O/R + L` shows this system is memory-bandwidth-bound: "
            f"weights must be read from HBM for every generated token. "
            f"Fix: add H100 GPUs (throughput scales linearly), "
            f"switch to continuous batching (1.0× vs 0.5× throughput multiplier), "
            f"or increase quantization level to reduce D (bytes accessed)."
        ), kind="danger"))

    if not slo_cost_ok:
        _banners.append(mo.callout(mo.md(
            f"**Cost SLO Violated.** "
            f"Actual cost: **${cost_per_req_usd:.5f}/req** exceeds the **${SLO_COST_PER_REQ}/req** SLO. "
            f"Cost model: `cost/req = (N_GPUs × $/hr) / (req/hr capacity)`. "
            f"Fix: increase quantization (fewer bytes/param → more tokens/sec → lower cost/req), "
            f"or switch to continuous batching to maximize GPU utilization."
        ), kind="danger"))

    if effective_accuracy < 0.85:
        _banners.append(mo.callout(mo.md(
            f"**Accuracy Degradation Warning.** "
            f"Selected quantization levels produce a combined accuracy of **{effective_accuracy * 100:.1f}%** "
            f"before drift. Ternary quantization on the mobile tier loses ~12% accuracy — "
            f"acceptable for a trigger/routing model but not for primary inference. "
            f"Ensure your tier routing sends precision-sensitive requests to the cloud tier."
        ), kind="warn"))

    if post_drift_accuracy < 0.88:
        _banners.append(mo.callout(mo.md(
            f"**Drift Accumulation Warning.** "
            f"At a {_cadence}-day retraining cadence, post-drift accuracy falls to "
            f"**{post_drift_accuracy * 100:.1f}%** (from {effective_accuracy * 100:.1f}% baseline). "
            f"The degradation equation `Acc(t) = Acc_0 − λ × PSI(t)` predicts this decay. "
            f"Fix: shorten retraining cadence or implement PSI monitoring with automated triggers."
        ), kind="warn"))

    if not _banners:
        mo.callout(mo.md(
            "**All constraints satisfied.** Your system design is physically feasible "
            "and meets all SLOs. The architecture diagram above shows a valid full-stack "
            "deployment. Review the key takeaways to consolidate your understanding."
        ), kind="success")
    else:
        mo.vstack(_banners)
    return


# ── CELL 20: ARCHITECTURE DIAGRAM ─────────────────────────────────────────────
@app.cell(hide_code=True)
def _(
    mo, go, COLORS, apply_plotly_theme,
    cloud_model_gb, edge_model_gb, mobile_model_gb, tiny_model_kb,
    cloud_p99_ms, edge_p99_ms, mobile_p99_ms,
    cloud_oom, edge_oom, mobile_oom, tiny_oom,
    slo_latency_ok, slo_cost_ok,
    cost_per_req_usd, effective_accuracy,
    _n_gpus, _qc, _qe, _qm,
    H100_RAM_GB, ORIN_RAM_GB, MOBILE_RAM_GB, MCU_SRAM_KB,
    SLO_P99_MS, SLO_COST_PER_REQ,
):
    # Architecture: 4 tiers plotted as bubbles on Memory (x) vs Latency (y) chart
    # Bubble size = model footprint. Color = feasibility.

    _tier_data = [
        {
            "name":    f"Cloud (H100×{_n_gpus}, {_qc.upper()})",
            "x":       cloud_model_gb,
            "y":       cloud_p99_ms,
            "oom":     cloud_oom,
            "ram":     _n_gpus * H100_RAM_GB,
            "color":   COLORS["Cloud"],
            "size":    40,
        },
        {
            "name":    f"Edge (Orin, {_qe.upper()})",
            "x":       edge_model_gb,
            "y":       edge_p99_ms,
            "oom":     edge_oom,
            "ram":     ORIN_RAM_GB,
            "color":   COLORS["Edge"],
            "size":    32,
        },
        {
            "name":    f"Mobile (NPU, {_qm.upper()})",
            "x":       mobile_model_gb,
            "y":       mobile_p99_ms,
            "oom":     mobile_oom,
            "ram":     MOBILE_RAM_GB,
            "color":   COLORS["Mobile"],
            "size":    28,
        },
        {
            "name":    "TinyML (Cortex-M7, Ternary, 10M sub-model)",
            "x":       tiny_model_kb / 1024,   # convert KB → GB for consistent axis
            "y":       8000.0,                  # TinyML latency is orders of magnitude higher
            "oom":     tiny_oom,
            "ram":     MCU_SRAM_KB / (1024 * 1024),  # 256 KB → GB for display
            "color":   COLORS["Tiny"],
            "size":    20,
        },
    ]

    _fig = go.Figure()

    # SLO region (green background below P99 = 100 ms)
    _fig.add_shape(
        type="rect",
        x0=0, x1=100, y0=0, y1=SLO_P99_MS,
        fillcolor="rgba(0,143,69,0.06)",
        line=dict(color=COLORS["GreenLine"], width=1, dash="dot"),
        layer="below",
    )
    _fig.add_annotation(
        x=80, y=SLO_P99_MS * 0.5,
        text="SLO region (P99 < 100 ms)",
        showarrow=False,
        font=dict(size=9, color=COLORS["GreenLine"]),
    )

    for _t in _tier_data:
        _marker_color = (
            COLORS["RedLine"] if _t["oom"]
            else _t["color"]
        )
        _symbol = "x" if _t["oom"] else "circle"
        _fig.add_trace(go.Scatter(
            x=[_t["x"]],
            y=[_t["y"]],
            mode="markers+text",
            name=_t["name"],
            marker=dict(
                size=_t["size"],
                color=_marker_color,
                symbol=_symbol,
                line=dict(color="white", width=2),
                opacity=0.85 if not _t["oom"] else 1.0,
            ),
            text=[_t["name"].split(" (")[0]],
            textposition="top center",
            textfont=dict(size=9, color=COLORS["Text"]),
        ))

        # RAM budget line: vertical dashed line at RAM capacity
        _fig.add_shape(
            type="line",
            x0=_t["ram"], x1=_t["ram"],
            y0=0, y1=1,
            yref="paper",
            line=dict(color=_t["color"], width=1.5, dash="dot"),
        )

    _fig.update_layout(
        height=380,
        showlegend=True,
        legend=dict(orientation="h", y=-0.18, font=dict(size=9)),
        xaxis=dict(
            title="Model Memory Footprint (GB)",
            type="log",
            range=[-4, 2],
            gridcolor="#f1f5f9",
        ),
        yaxis=dict(
            title="P99 Latency (ms, log scale)",
            type="log",
            range=[0, 5],
            gridcolor="#f1f5f9",
        ),
        title=dict(
            text="4-Tier System Architecture — Memory vs Latency (dashed lines = RAM budgets)",
            font=dict(size=11, color=COLORS["TextSec"]),
            x=0.5,
        ),
    )
    apply_plotly_theme(_fig)

    mo.vstack([
        mo.md("### System Architecture — Memory vs Latency Map"),
        mo.md(
            "Each tier is plotted at its model footprint (x-axis) and P99 latency (y-axis). "
            "Dashed vertical lines show each tier's RAM budget. "
            "A tier marked with × has exceeded its memory budget (OOM). "
            "The green shaded region is the P99 SLO zone."
        ),
        mo.ui.plotly(_fig),
    ])
    return


# ── CELL 21: ACT II PREDICTION REVEAL ────────────────────────────────────────
@app.cell(hide_code=True)
def _(
    mo,
    act2_prediction,
    cloud_oom, edge_oom, mobile_oom, tiny_oom,
    slo_latency_ok, slo_cost_ok,
    cloud_model_gb, edge_model_gb, mobile_model_gb, tiny_model_kb,
    H100_RAM_GB, ORIN_RAM_GB, MOBILE_RAM_GB, MCU_SRAM_KB,
):
    _chosen  = act2_prediction.value

    # Determine which constraint is actually hardest
    # "hardest" = the one that fails first with default settings
    # With defaults: 2×H100, INT8 cloud, INT8 edge, INT4 mobile
    _default_cloud_oom  = (7.0 + 1.0) > 160   # 8 GB model vs 160 GB  → False
    _default_edge_oom   = (7.0 + 1.0) > 16    # 8 GB model vs 16 GB   → False (just barely)
    _default_mobile_oom = (3.5 + 0.5) > 8     # 4 GB model vs 8 GB    → False
    _default_tiny_oom   = (10e6 * 0.19 * 1024 / 1e6) > 256  # ~1900 KB > 256 KB → True

    # The TinyML tier is the actual hardest constraint: even the 10M sub-model
    # at ternary requires ~1.9 MB, which exceeds 256 KB SRAM by 7.5×.
    # Students must use a much smaller keyword-trigger model (< 100K params).
    _actual_hardest = "tiny_oom"
    _actual_correct = (_chosen == _actual_hardest)

    _explanations = {
        "cloud_oom": (
            "**Good intuition about memory, but the cloud tier is actually the most forgiving.** "
            "With tensor-parallel sharding across N H100s, each with 80 GB HBM, the cloud tier "
            "can handle the full 7B model even at FP16 (14 GB) with just 1 GPU. "
            "The real cloud constraint is *latency* and *cost*, not memory."
        ),
        "edge_latency": (
            "**Edge latency is a real concern, but not the primary failure mode.** "
            "Jetson Orin NX has 16 GB RAM — the 7B model at INT8 (7 GB) fits with margin. "
            "Latency is high for full-sequence generation, but edge tier can be scoped "
            "to shorter responses or used as a cache/filter. "
            "The harder constraint is the tier that has the least room: TinyML."
        ),
        "mobile_oom": (
            "**Mobile OOM is a genuine risk, but it's manageable.** "
            "The 7B model at FP16 (14 GB) absolutely won't fit in an 8 GB smartphone. "
            "But at INT4 (3.5 GB + ~0.5 GB KV cache), it fits. "
            "The mobile tier is a real constraint — but it has an engineering solution. "
            "The *impossible* constraint is TinyML SRAM."
        ),
        "tiny_oom": (
            "**Correct — TinyML is the hardest constraint, and it is fundamentally different "
            "from the other tiers.** "
            "Even a 10M-parameter ternary sub-model requires ~1.9 MB of SRAM — 7.5× more "
            "than the Cortex-M7's 256 KB ceiling. This is not a quantization problem you "
            "can engineer away with better precision. The TinyML tier requires a completely "
            "different *model architecture*: a 50–100K parameter keyword-spotting or anomaly "
            "detection model, not a miniaturized LLM. The 256 KB constraint is a wall, "
            "not a parameter to tune."
        ),
    }

    _explanation = _explanations.get(_chosen or "tiny_oom", "")

    mo.callout(
        mo.md(
            f"**{'Correct.' if _actual_correct else 'Reassessing.'}** {_explanation}"
        ),
        kind="success" if _actual_correct else "warn",
    )
    return


# ── CELL 22: ACT II MATHPEEK ──────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "Full-Stack Constraint Synthesis — All 7 Invariants": mo.md("""
        The Act II design challenge activates all 7 invariants simultaneously.
        Here is how each appears in the 7B LLM deployment scenario:

        **1. Iron Law → Cloud P99 Latency**
        ```
        T_cloud = output_tokens / (BW_per_GPU × N_GPUs / bytes_per_param × serving_mult)
        ```
        For 256 output tokens on H100×2 with INT8 continuous batching:
        T ≈ 256 / (3350 GB/s × 2 / 1 byte × 1.0) ≈ 0.038 ms per token → ~9.7 ms total

        **2. Memory Wall → Tier Feasibility**
        ```
        Tier_feasible = (model_GB + KV_cache_GB) ≤ tier_RAM_GB
        ```
        Cloud (FP16): 14 GB + 2 GB = 16 GB. Fits on 1× H100 (80 GB). ✓
        TinyML (Ternary, 10M params): 1.9 MB >> 256 KB SRAM. ✗

        **3. Amdahl's Law → Scaling Ceiling**
        ```
        Speedup(N_GPUs) = 1 / (S + (1-S)/N)
        ```
        Communication overhead between GPUs introduces a serial fraction S ≈ 0.05–0.10.
        Speedup saturates at 10–20× regardless of GPU count.

        **4. Roofline → Memory vs Compute Bound**
        ```
        AI* = Peak_FLOPS / Memory_BW
        7B model AI ≈ 1 op/byte → well below ridge point → memory-bound
        ```
        At decode time, the 7B model is almost always memory-bandwidth-bound.
        Adding FLOPs does not help.

        **5. Little's Law → Serving Utilization**
        ```
        L = λ × W   →   at λ = 1000 req/s, W = 0.01 s → L = 10 requests in queue
        ```
        Utilization ρ = λ/μ. At ρ > 0.8, P99 diverges from average. Continuous batching
        keeps ρ lower by processing tokens in parallel across requests.

        **6. Degradation Equation → Retraining Cadence**
        ```
        Acc(t) = Acc_0 - λ_drift × PSI(t)    λ_drift ≈ 0.001/day
        ```
        At 30-day cadence: Acc(30) = Acc_0 - 0.03. At 90-day: Acc_0 - 0.09.
        Mobile and Edge tiers drift differently — they need tier-specific monitoring.

        **7. Fairness-Accuracy Tradeoff**
        ```
        Fairness_gap ≤ threshold  →  requires accuracy sacrifice ∝ base_rate_difference
        ```
        A tighter fairness threshold (e.g., < 5% gap) costs more accuracy on average
        than a looser threshold (< 20% gap). This cost is distributed unevenly across
        demographic groups.
        """),
    })
    return


# ── CELL 23: ACT II REFLECTION ────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(
    mo,
    system_valid, constraint_hit,
    n_h100, quant_cloud, quant_edge, quant_mobile,
    serving_strategy, retrain_cadence_days,
    cloud_p99_ms, cost_per_req_usd, effective_accuracy,
    SLO_P99_MS, SLO_COST_PER_REQ,
):
    _n      = n_h100.value
    _qc     = quant_cloud.value
    _qe     = quant_edge.value
    _qm     = quant_mobile.value
    _srv    = serving_strategy.value
    _cad    = retrain_cadence_days.value

    if system_valid:
        mo.callout(mo.md(f"""
**System design is feasible.** Your configuration satisfies all hard constraints.

**Configuration summary:**
- Cloud: {_n}× H100, {_qc.upper()} quantization, {_srv} serving
- Edge: {_qe.upper()} quantization
- Mobile: {_qm.upper()} quantization
- TinyML: ternary sub-model (10M params, fixed)
- Retraining cadence: {_cad} days

**Key metrics:**
- P99: {cloud_p99_ms:.1f} ms (SLO: {SLO_P99_MS} ms)
- Cost: ${cost_per_req_usd:.5f}/req (SLO: ${SLO_COST_PER_REQ})
- Accuracy: {effective_accuracy * 100:.1f}%

This is a valid architecture. In production, you would next specify:
monitoring cadence per tier, alerting thresholds, A/B rollout plan,
and the fallback routing policy when a lower tier is overloaded.
        """), kind="success")
    elif constraint_hit:
        mo.callout(mo.md(f"""
**One or more constraints are violated.** Review the failure state banners above.

The most common fix sequence:
1. **Edge OOM?** → switch edge quantization to INT8 or INT4.
2. **P99 SLO?** → switch to continuous batching first, then add H100s.
3. **Cost SLO?** → increase quantization level (fewer bytes → more tokens/sec → lower cost/req).
4. **TinyML OOM?** → the 10M-param sub-model is fixed in this scenario; in practice,
   you would replace the TinyML tier with a purpose-built 50K-param classifier.

Each fix is directly traceable to one of the 7 invariants. There is no free lunch.
        """), kind="warn")
    else:
        mo.callout(mo.md("Adjust the controls above to configure the system."), kind="info")
    return


# ─────────────────────────────────────────────────────────────────────────────
# ═════════════════════════════════════════════════════════════════════════════
# ZONE D: CLOSING
# ═════════════════════════════════════════════════════════════════════════════

# ─── CELL 20: SYNTHESIS (CAPSTONE) ────────────────────────────────────────────
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
                Volume I Key Takeaways
            </div>
            <div style="font-size: 0.92rem; color: {COLORS['Text']}; line-height: 1.75;">
                <div style="margin-bottom: 10px;">
                    <strong>1. Tier boundaries are physics constraints, not software choices.</strong>
                    The 256 KB SRAM ceiling on a Cortex-M7 is immovable. The 5-watt thermal budget
                    on a smartphone is immovable. Each tier boundary is set by thermodynamics, memory
                    physics, and signal latency &mdash; not by framework version or quantization
                    scheme. The architect&apos;s first job is to understand which constraints are
                    negotiable and which are walls.
                </div>
                <div style="margin-bottom: 10px;">
                    <strong>2. All 7 invariants are active simultaneously in production.</strong>
                    The Iron Law governs cloud P99. Amdahl caps GPU scaling. The Memory Wall
                    determines tier assignment. Little&apos;s Law predicts P99 explosion at
                    utilization. The degradation equation sets retraining cadence. Chouldechova
                    bounds the fairness-accuracy tradeoff. Every slider you move shifts multiple
                    constraints at once; no single optimization is ever free.
                </div>
                <div>
                    <strong>3. Complexity cannot be destroyed &mdash; only moved.</strong>
                    Every sub-team hitting its individual target is a necessary but insufficient
                    condition for system success. The integrated system failed because INT8
                    quantization shifted complexity from the model to the monitoring and skew
                    pipeline &mdash; a pipeline no sub-team owned. Systems thinking is the
                    invariant that governs all other invariants.
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
                    What's Next &mdash; Volume II
                </div>
                <div style="font-size: 0.88rem; color: {COLORS['TextSec']}; line-height: 1.6;">
                    Volume I covered single-machine ML systems (1&ndash;8 GPUs). Volume II asks:
                    what happens when the same constraints &mdash; the Iron Law, Amdahl, Little&apos;s
                    Law, drift &mdash; operate across hundreds of machines, connected by a network
                    fabric that becomes the new memory bandwidth bottleneck? The binding constraint
                    you found in this capstone will appear again &mdash; at rack scale.
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
                    <strong>Read:</strong> @sec-conclusion-synthesis for the full seven-invariant
                    framework and @sec-conclusion-design-philosophy for the systems-first
                    perspective that unifies all Volume I chapters.<br/>
                    <strong>Build:</strong> TinyTorch Module 16 &mdash; the capstone integration,
                    combining profiling, serving, drift detection, and fairness auditing into a
                    single end-to-end pipeline. See <code>tinytorch/src/16_capstone/</code>.
                </div>
            </div>

        </div>
        """),

        mo.accordion({
            "Self-Assessment: Can you answer these?": mo.md("""
    1. Llama-2-70B (FP16) on H100 has T_mem approximately 42 ms and T_comp approximately 0.14 ms for a single token of autoregressive decoding. What is the memory-to-compute ratio — and why does optimizing CUDA kernel utilization from 70% to 95% yield less than 1% end-to-end speedup?

    2. The Conservation of Complexity states that complexity cannot be destroyed, only moved between Data, Algorithm, and Machine. In the Act II MobileNetV2 deployment, switching from FP16 to INT8 reduces memory complexity but increases which other complexity dimension — and what is the Iron Law invariant that formalizes this interaction?

    3. A team where every sub-team hit its own metric (efficient architecture, 4x compression, P99 < 50 ms) still shipped a system that lost 4 accuracy points within weeks. Identify the specific cross-invariant interaction that caused the silent failure — and explain why monitoring infrastructure metrics (uptime, latency, error rate) could not have detected it.

    *If you cannot answer all three from memory, revisit Act I and Act II.*
    """)
        }),
    ])
    return


# ─── CELL 21: LEDGER SAVE + HUD ──────────────────────────────────────────────
@app.cell(hide_code=True)
def _(
    mo, ledger, COLORS,
    act1_prediction, act1_correct_answer,
    act2_prediction,
    n_h100, quant_cloud, quant_edge, quant_mobile,
    serving_strategy, retrain_cadence_days, fairness_threshold_pct,
    cloud_p99_ms, cost_per_req_usd, effective_accuracy,
    tier_assignments, system_valid, constraint_hit,
    n_completed, pct_correct, weakest_invariant,
):
    # ── Resolve values with safe defaults ─────────────────────────────────────
    _a1_pred   = act1_prediction.value  or "none"
    _a1_ok     = (_a1_pred == act1_correct_answer)
    _a2_pred   = act2_prediction.value  or "none"
    _a2_ok     = (_a2_pred == "tiny_oom")  # correct answer for act2

    _n         = n_h100.value
    _qc        = quant_cloud.value
    _qe        = quant_edge.value
    _qm        = quant_mobile.value
    _srv       = serving_strategy.value
    _cad       = retrain_cadence_days.value
    _gap_thr   = fairness_threshold_pct.value

    _p99       = float(cloud_p99_ms) if cloud_p99_ms is not None else 0.0
    _cost      = float(cost_per_req_usd) if cost_per_req_usd is not None else 0.0
    _acc       = float(effective_accuracy) if effective_accuracy is not None else 0.0

    ledger.save(
        chapter=16,
        design={
            "context":                "all",
            "cloud_hardware":         f"H100x{_n}",
            "quantization_strategy":  _qc,
            "serving_strategy":       _srv,
            "total_cost_per_req":     round(_cost, 6),
            "p99_latency_ms":         round(_p99, 2),
            "tier_assignments":       tier_assignments,
            "retraining_cadence_days": int(_cad),
            "fairness_threshold_pct": int(_gap_thr),
            "act1_prediction":        _a1_pred,
            "act1_correct":           _a1_ok,
            "act2_prediction":        _a2_pred,
            "act2_correct":           _a2_ok,
            "act2_result":            "feasible" if system_valid else "infeasible",
            "act2_decision":          f"H100x{_n}_{_qc}_{_srv}",
            "constraint_hit":         bool(constraint_hit),
            "system_valid":           bool(system_valid),
            "labs_completed":         int(n_completed),
            "pct_correct":            round(float(pct_correct), 1),
            "weakest_invariant":      weakest_invariant,
        },
    )

    # ── HUD ──────────────────────────────────────────────────────────────────
    _p99_ok   = _p99 <= 100.0
    _cost_ok  = _cost <= 0.001
    _sys_ok   = bool(system_valid)

    _p99_color  = "#4ade80" if _p99_ok  else "#f87171"
    _cost_color = "#4ade80" if _cost_ok else "#f87171"
    _sys_color  = "#4ade80" if _sys_ok  else "#f87171"
    _a1_color   = "#4ade80" if _a1_ok   else "#f87171"
    _a2_color   = "#4ade80" if _a2_ok   else "#f87171"
    _pct_color  = "#4ade80" if pct_correct >= 70 else ("#fbbf24" if pct_correct >= 45 else "#f87171")

    mo.Html(f"""
    <div style="display:flex; gap:22px; align-items:center; flex-wrap:wrap;
                padding:14px 24px; background:#0f172a;
                border-radius:12px; margin-top:32px; font-size:0.79rem;
                border:1px solid #1e293b; font-family:'SF Mono',monospace;">
        <span style="color:#475569; font-weight:700; letter-spacing:0.06em; font-size:0.82rem;">
            LAB 16 &mdash; CAPSTONE
        </span>
        <span>
            <span style="color:#475569;">LABS DONE </span>
            <span style="color:#e2e8f0; font-weight:700;">{n_completed}/15</span>
        </span>
        <span>
            <span style="color:#475569;">PREDICTION ACC </span>
            <span style="color:{_pct_color}; font-weight:700;">{pct_correct:.0f}%</span>
        </span>
        <span>
            <span style="color:#475569;">WEAKEST INV </span>
            <span style="color:{COLORS['OrangeLine']}; font-weight:700;">{weakest_invariant}</span>
        </span>
        <span>
            <span style="color:#475569;">ACT I </span>
            <span style="color:{_a1_color};">{"Correct" if _a1_ok else ("Incorrect" if _a1_pred != "none" else "—")}</span>
        </span>
        <span>
            <span style="color:#475569;">ACT II PRED </span>
            <span style="color:{_a2_color};">{"Correct" if _a2_ok else ("Incorrect" if _a2_pred != "none" else "—")}</span>
        </span>
        <span>
            <span style="color:#475569;">P99 </span>
            <span style="color:{_p99_color}; font-weight:700;">{_p99:.1f} ms</span>
        </span>
        <span>
            <span style="color:#475569;">COST </span>
            <span style="color:{_cost_color}; font-weight:700;">${_cost:.5f}</span>
        </span>
        <span>
            <span style="color:#475569;">SYSTEM </span>
            <span style="color:{_sys_color}; font-weight:700;">{"VALID" if _sys_ok else "INFEASIBLE"}</span>
        </span>
        <span>
            <span style="color:#475569;">LEDGER </span>
            <span style="color:#4ade80;">ch16 saved &mdash; Vol I complete</span>
        </span>
    </div>
    """)
    return


if __name__ == "__main__":
    app.run()

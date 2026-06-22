import marimo

__generated_with = "0.23.1"
app = marimo.App(width="full")

# ─────────────────────────────────────────────────────────────────────────────
# LAB 00: THE ARCHITECT'S PORTAL
#
# This is an ML Systems lab — not an ML lab.
# Students are not here to learn how models work.
# They are here to understand that where a model runs
# determines whether it can run at all.
#
# Four concept modules:
#   Part A. Deployed behavior, not isolated models
#   Part B. Track as operating envelope
#   Part C. Repeated case -> prediction -> manipulation -> evidence -> decision -> report ritual
#   Part D. Ledger continuity and local report handoff
# No physics instruments (introduced in Lab 01+).
# No prediction locks in anger (students haven't read Chapter 1 yet).
# Progressive disclosure: each check gates the next concept.
#
# Concepts covered (all from pre-reading context, no chapter required):
#   1. ML systems labs are about deployed behavior, not isolated models.
#   2. A track is an operating envelope, not a career choice.
#   3. Later labs repeat a case -> prediction -> manipulation -> evidence -> decision -> report loop.
#   4. The report unlocks only after the orientation and track choice are complete.
#
# Design Ledger: initialized with deployment context at completion.
# ─────────────────────────────────────────────────────────────────────────────

# ═══════════════════════════════════════════════════════════════════════════════
# ZONE A: OPENING
# ═══════════════════════════════════════════════════════════════════════════════

# ─── CELL 0: SETUP ─────────────────────────────────────────────────────────────
@app.cell
async def _():
    import marimo as mo
    import sys
    from pathlib import Path

    if sys.platform == "emscripten":
        import micropip
        await micropip.install(["pydantic", "pint", "plotly", "pandas"], keep_going=False)
        await micropip.install("../../wheels/mlsysim-0.1.2-py3-none-any.whl", keep_going=False)
        await micropip.install("../../wheels/mlsysbook_labs-0.1.0-py3-none-any.whl", keep_going=False)
    else:
        _labs_dir = Path(__file__).resolve().parents[1]
        if str(_labs_dir) not in sys.path:
            sys.path.insert(0, str(_labs_dir))
        from bootstrap import native_bootstrap
        native_bootstrap(__file__)

    import plotly.graph_objects as go
    from mlsysim.labs.state import DesignLedger
    from mlsysim.labs.style import COLORS, LAB_CSS
    from mlsysim.labs.components import DecisionLog
    from mlsysbook_labs import (
        ACADEMIC_LAB_CSS,
        CANONICAL_TRACKS,
        LabMetadata,
        build_lab_report,
        get_track_profile,
        report_export_panel,
        track_context,
        track_arc_context,
        track_display_label,
    )

    ledger = DesignLedger()
    if getattr(ledger, "is_wasm", False):
        _ = await ledger.load_async()
    return (
        COLORS,
        ACADEMIC_LAB_CSS,
        CANONICAL_TRACKS,
        DecisionLog,
        LAB_CSS,
        LabMetadata,
        build_lab_report,
        get_track_profile,
        ledger,
        mo,
        report_export_panel,
        track_context,
        track_arc_context,
        track_display_label,
    )

@app.cell
def _(LabMetadata):
    lab_metadata = LabMetadata(
        lab_id="v1_00_architects_portal",
        title="The Architect's Portal",
        volume="Volume I",
        chapter="Orientation",
        book_anchor="Volume I orientation",
        lab_version="1.0.0",
        updated_at="2026-06-03",
        release_channel="dev",
        mlsysim_version="0.1.2",
    )
    lab_learning_objectives = (
        "Explain what an ML systems lab is trying to teach.",
        "Compare the four canonical tracks and choose one recurring operating envelope.",
        "Recognize the repeated lab rhythm: case, prediction, manipulation, evidence, decision, and report.",
    )
    lab_big_takeaways = (
        "The same model idea becomes a different systems problem on 📱 iPhone, 💍 Oura Ring, 🚕 RoboTaxi, and ☁️ Cloud Fleet.",
        "Track selection changes hardware facts, constraints, metrics, stakeholder pressure, failure modes, and report framing.",
        "Later labs build on this track choice, but each lab will introduce only the new concept it needs.",
    )
    return lab_big_takeaways, lab_learning_objectives, lab_metadata

@app.cell
def _():
    def v1_00_track_story(track_id):
        stories = {
            "iphone": {
                "persona": "UX director",
                "stakeholder_question": "Can this run locally without making the phone hot, slow, or privacy-invasive?",
                "first_bottleneck": "thermal envelope",
                "likely_failure": "thermal throttle, battery drain, or sluggish interactive UX",
                "report_frame": "local-device readiness memo: responsiveness, privacy, battery, and sustained comfort",
                "amount_reasoning": "milliseconds and joules must fit a handheld user experience",
            },
            "oura_ring": {
                "persona": "hardware lead",
                "stakeholder_question": "Can the sensing window, model, buffers, and OTA package fit inside a tiny wearable budget?",
                "first_bottleneck": "SRAM",
                "likely_failure": "SRAM or flash overflow, duty-cycle violation, or radio wakeup budget miss",
                "report_frame": "firmware fit memo: memory, update size, sensing cadence, and battery life",
                "amount_reasoning": "bytes and microjoules decide whether the feature can stay always on",
            },
            "robotaxi": {
                "persona": "safety director",
                "stakeholder_question": "Can perception meet the deadline and protect rare-event safety margins?",
                "first_bottleneck": "tail latency",
                "likely_failure": "p99/p999 deadline miss, safety-margin miss, or unsupported fallback path",
                "report_frame": "safety evidence memo: worst-case latency, rare-event recall, and fallback plan",
                "amount_reasoning": "tail milliseconds and missed detections become safety evidence",
            },
            "cloud_fleet": {
                "persona": "CTO",
                "stakeholder_question": "Can the service satisfy demand without wasting capacity, money, or carbon budget?",
                "first_bottleneck": "utilization",
                "likely_failure": "SLO breach, queue growth, negative ROI, or carbon budget miss",
                "report_frame": "fleet operations memo: SLA, throughput, utilization, cost/request, and carbon",
                "amount_reasoning": "requests, dollars, utilization points, and carbon turn scale into an operating constraint",
            },
        }
        return stories[track_id]

    return (v1_00_track_story,)

# ─── CELL 1: HEADER ────────────────────────────────────────────────────────────
@app.cell
def _(ACADEMIC_LAB_CSS, LAB_CSS, lab_metadata, mo):
    mo.vstack([
        LAB_CSS,
        ACADEMIC_LAB_CSS,
        mo.md(f"""
        <div style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
                    padding: 36px 44px; border-radius: 16px; color: white;
                    box-shadow: 0 8px 32px rgba(0,0,0,0.3);">
            <div style="font-size: 0.72rem; font-weight: 700; letter-spacing: 0.18em;
                        color: #475569; text-transform: uppercase; margin-bottom: 10px;">
                Machine Learning Systems · Volume I · Lab 00
            </div>
            <h1 style="margin: 0 0 10px 0; font-size: 2.4rem; font-weight: 900;
                       color: #f8fafc; line-height: 1.1; letter-spacing: -0.02em;">
                {lab_metadata.title}
            </h1>
            <p style="margin: 0 0 20px 0; font-size: 1.05rem; color: #94a3b8;
                      max-width: 620px; line-height: 1.65;">
                This orientation shows how the lab sequence works: choose a track,
                learn the repeated workflow, and see how later labs adapt the same
                concept to your selected context.
            </p>
            <div style="display: flex; gap: 12px; flex-wrap: wrap;">
                <span style="background: rgba(99,102,241,0.15); color: #a5b4fc;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(99,102,241,0.25);">
                    Orientation · Track Choice · Interface Tour
                </span>
                <span style="background: rgba(16,185,129,0.15); color: #6ee7b7;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(16,185,129,0.25);">
                    Lab v{lab_metadata.lab_version}
                </span>
                <span style="background: rgba(245,158,11,0.15); color: #fcd34d;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(245,158,11,0.25);">
                    Local report artifact
                </span>
            </div>
        </div>
        """),
    ])
    return

# ─── CELL 2: BRIEFING ──────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(COLORS, lab_learning_objectives, mo):
    _objective_rows = "".join(
        f"""<div style="margin-bottom: 3px;">{idx}. <strong>{objective}</strong></div>"""
        for idx, objective in enumerate(lab_learning_objectives, start=1)
    )
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
                {_objective_rows}
            </div>
        </div>

        <div style="border-top: 1px solid {COLORS['Border']}; margin: 0 -28px; padding: 0 28px;"></div>

        <!-- CHAPTER RECAP -->
        <div style="margin-top: 16px; margin-bottom: 16px;">
            <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                        text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                Chapter Recap
            </div>
            <div style="font-size: 0.88rem; color: {COLORS['TextSec']}; line-height: 1.65;">
                Orientation introduces the central MLSys idea: a model becomes a system only when
                it runs inside a concrete machine, workload, stakeholder context, and physical
                constraint budget. The common trap is treating deployment as a late-stage detail.
            </div>
        </div>

        <div style="border-top: 1px solid {COLORS['Border']}; margin: 0 -28px; padding: 0 28px;"></div>

        <!-- PREREQUISITES + DURATION -->
        <div style="display: flex; gap: 32px; margin-top: 16px; margin-bottom: 16px; flex-wrap: wrap;">
            <div style="flex: 1; min-width: 220px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                    Prerequisites
                </div>
                <div style="font-size: 0.85rem; color: {COLORS['TextSec']}; line-height: 1.65;">
                    No prior reading required &mdash; this lab introduces the curriculum
                    from first principles. Concepts here will be reinforced in
                    Chapter 1: Introduction and Chapter 2: ML Systems.
                </div>
            </div>
            <div style="flex: 0 0 180px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                    Duration
                </div>
                <div style="font-size: 0.85rem; color: {COLORS['TextSec']}; line-height: 1.65;">
                    <strong>20&ndash;25 min</strong><br/>
                    Orientation Checks &middot; Interface Tour
                </div>
            </div>
        </div>

        <div style="border-top: 1px solid {COLORS['Border']}; margin: 0 -28px; padding: 0 28px;"></div>

        <!-- LAB MAP -->
        <div style="margin-top: 16px; margin-bottom: 16px;">
            <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                        text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                Lab Map
            </div>
            <div style="font-size: 0.86rem; color: {COLORS['TextSec']}; line-height: 1.7;">
                <strong>Part A:</strong> deployed behavior, not isolated models &middot;
                <strong>Part B:</strong> track as operating envelope &middot;
                <strong>Part C:</strong> case, prediction, manipulation, evidence, decision, report &middot;
                <strong>Part D:</strong> ledger/report continuity &middot;
                <strong>Synthesis:</strong> explain how your track shapes later amount-system reasoning.
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
                "What kind of work will these labs ask you to do, and how will your
                chosen track change the story without changing the core lesson?"
            </div>
        </div>
    </div>
    """)
    return

# ═══════════════════════════════════════════════════════════════════════════════
# ZONE B: ORIENTATION CHECKS
# ═══════════════════════════════════════════════════════════════════════════════

# ─── ORIENTATION 1: WHAT THIS COURSE MEANS ─────────────────────────────────────
# _act_why: "Students should know what kind of work these labs ask them to do before any technical scenario appears."
@app.cell
def _(mo):
    mo.vstack([
        mo.md("---"),
        mo.md("""
        ## Part A - Deployed Behavior, Not Isolated Models

        **Scenario.** A model demo works in a notebook. Your lab lead asks whether
        it is ready to become a deployed ML system. The first mistake is to answer
        using only the model score.

        These labs are not a second textbook and they are not a quiz bank. They are a
        repeated way to practice systems thinking around machine learning: make a
        prediction, inspect deployed evidence, name the constraint, and write the
        decision you would defend.
        """),
        mo.Html("""
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin: 16px 0;">
            <div style="background: #fef2f2; border: 1px solid #fecaca; border-radius: 12px;
                        padding: 20px; border-left: 5px solid #ef4444;">
                <div style="font-weight: 800; color: #991b1b; margin-bottom: 8px;">
                    A Model In Isolation
                </div>
                <div style="color: #7f1d1d; font-size: 0.9rem; line-height: 1.6;">
                    You ask whether the model is accurate on a dataset. That matters,
                    but it is only one part of the work.
                </div>
            </div>
            <div style="background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 12px;
                        padding: 20px; border-left: 5px solid #16a34a;">
                <div style="font-weight: 800; color: #14532d; margin-bottom: 8px;">
                    A Model In A System
                </div>
                <div style="color: #14532d; font-size: 0.9rem; line-height: 1.6;">
                    You ask where the model runs, what it costs, how fast it must respond,
                    what can fail, and what evidence would make the design defensible.
                </div>
            </div>
        </div>
        """),
        mo.md("""
        **Consequence.** Notebook success is not deployment success. The chapter's
        opening claim is that ML systems have a physics: learned behavior must move
        through memory, consume energy, meet latency windows, and survive a real
        operating context.

        Lab 00 only orients you to that pattern. Lab 01 is where the first technical
        diagnosis begins.
        """),
        mo.accordion({
            "Reading connection": mo.md("""
            Chapter 1 frames ML systems as data-shaped behavior running under
            physical constraint. Part A turns that claim into the first lab habit:
            do not stop at the model; ask what deployed behavior the system must
            prove.
            """),
        }),
    ])
    return

# ─── CHECK 1 ───────────────────────────────────────────────────────────────────

@app.cell
def _(mo):
    check1 = mo.ui.radio(
        options={
            "A)  Memorize facts about four devices": "A",
            "B)  Tune model accuracy before anything else": "B",
            "C)  Practice a repeated workflow for making evidence-backed systems decisions": "C",
            "D)  Choose a permanent career specialization": "D",
        },
        label="""**Part A prediction.** What are these labs mainly trying to help you practice?""",
    )
    return (check1,)

@app.cell
def _(check1, mo):
    if check1.value is None:
        _check1_view = mo.vstack([
            check1,
            mo.callout(
                mo.md("_Select an answer to continue._"),
                kind="warn",
            ),
        ])
    else:
        _correct = check1.value == "C"
        _feedback = {
            "A": (
                "**Not quite.** The devices give each track a concrete story, but the goal "
                "is not memorization. The devices help you reason about constraints."
            ),
            "B": (
                "**Not quite.** Accuracy matters, but these labs ask what else must be true "
                "before an ML idea can become a reliable system."
            ),
            "C": (
                "**Correct.** You will repeatedly read a situation, make a prediction, "
                "manipulate controls, inspect evidence, decide what you would defend, "
                "and download a short report."
            ),
            "D": (
                "**Not quite.** A track is only a learning lens. You can understand all four "
                "tracks while developing deeper intuition for one recurring context."
            ),
        }

        _check1_view = mo.vstack([
            check1,
            mo.callout(
                mo.md(_feedback[check1.value]),
                kind="success" if _correct else "warn",
            ),
        ])
    _check1_view
    return

# ─── ORIENTATION 2: WHAT A TRACK CHANGES ──────────────────────────────────────

@app.cell
def _(check1, mo):
    mo.stop(check1.value is None)

    mo.vstack([
        mo.md("---"),
        mo.md("""
        ## Part B - A Track Is An Operating Envelope

        **Scenario.** Four students study the same MLSys concept, but each one has
        to defend the result to a different stakeholder. The answer should not sound
        the same for a phone app, a wearable firmware update, a safety-critical
        vehicle, and a cloud service.

        A track is the operating envelope you carry through the labs. It changes
        the story, device, stakeholder, primary metrics, guardrails, likely failure
        mode, and report framing. It does **not** change the chapter concept
        everyone is learning.

        You can think of it as four students solving the same kind of problem
        under four different sets of constraints:
        """),
        mo.Html("""
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin: 16px 0;">

            <div style="background: white; border: 1px solid #e2e8f0; border-radius: 12px;
                        padding: 18px; border-top: 4px solid #6366f1;">
                <div style="font-size: 1.4rem; margin-bottom: 6px;">⚡</div>
                <div style="font-weight: 800; color: #1e293b; font-size: 0.95rem; margin-bottom: 6px;">
                    Where It Runs
                </div>
                <div style="color: #64748b; font-size: 0.85rem; line-height: 1.5;">
                    Cloud fleet, robotaxi, phone, and wearable hardware create
                    different limits. The same technique can be useful in one
                    context and irrelevant in another.
                </div>
            </div>

            <div style="background: white; border: 1px solid #e2e8f0; border-radius: 12px;
                        padding: 18px; border-top: 4px solid #ef4444;">
                <div style="font-size: 1.4rem; margin-bottom: 6px;">🌡️</div>
                <div style="font-weight: 800; color: #1e293b; font-size: 0.95rem; margin-bottom: 6px;">
                    What You Optimize
                </div>
                <div style="color: #64748b; font-size: 0.85rem; line-height: 1.5;">
                    One track may care first about cost, another about latency,
                    another about battery, and another about memory. The report
                    should use the track's metrics.
                </div>
            </div>

            <div style="background: white; border: 1px solid #e2e8f0; border-radius: 12px;
                        padding: 18px; border-top: 4px solid #10b981;">
                <div style="font-size: 1.4rem; margin-bottom: 6px;">💾</div>
                <div style="font-weight: 800; color: #1e293b; font-size: 0.95rem; margin-bottom: 6px;">
                    What You Defend
                </div>
                <div style="color: #64748b; font-size: 0.85rem; line-height: 1.5;">
                    The final decision should sound like it belongs to the
                    selected track's stakeholder. That keeps the lab from being
                    a generic worksheet.
                </div>
            </div>

        </div>
        """),
        mo.md("""
        **Consequence.** If a track only changes the label, it is not doing useful
        work. The track should change what constraint pushes back when the same
        idea is deployed.
        """),
    ])
    return

# ─── CHECK 2 (multi-select) ────────────────────────────────────────────────────

# Pattern C: widget definitions are ungated so they are always defined at
# module load. Downstream helpers (check2empty, check2value_list) and render
# cells depend on these being globally available even before check1 is answered.
@app.cell
def _(mo):
    model_size = mo.ui.checkbox(
        value=False,
        label="The story and stakeholder voice"
    )

    quantization = mo.ui.checkbox(
        value=False,
        label="The device and hardware assumptions"
    )

    move_server = mo.ui.checkbox(
        value=False,
        label="The primary metric and guardrails"
    )

    faster_gpu = mo.ui.checkbox(
        value=False,
        label="The report framing"
    )

    edge_deploy = mo.ui.checkbox(
        value=False,
        label="The chapter's core learning objective"
    )
    return (edge_deploy, faster_gpu, model_size, move_server, quantization)

@app.cell
def _(
    check1,
    check2empty,
    edge_deploy,
    faster_gpu,
    mo,
    model_size,
    move_server,
    quantization,
):
    mo.stop(check1.value is None)

    if check2empty():
        _check2_prompt = mo.vstack([
            mo.md("""**Part B prediction.** When you choose a track, which parts of
        later labs should change automatically?"""),

            mo.md("""Select **all** that should change because of the track:"""),
            model_size,
            quantization,
            move_server,
            faster_gpu,
            edge_deploy
        ])
    else:
        _check2_prompt = mo.md("")
    _check2_prompt
    return

@app.cell
def _(
    check1,
    check2empty,
    edge_deploy,
    faster_gpu,
    mo,
    model_size,
    move_server,
    quantization,
):
    mo.stop(check1.value is None or check2empty())

    _correct_set = {'model_size', 'quantization', 'move_server', 'faster_gpu'}
    _has_wrong     = edge_deploy.value
    _missing_right = not (model_size.value and quantization.value and move_server.value and faster_gpu.value)
    _exactly_right = not _has_wrong and not _missing_right

    _option_labels = {
        'model_size':   "Story and stakeholder voice",
        'quantization': "Device and hardware assumptions",
        'move_server':  "Primary metric and guardrails",
        'faster_gpu':   "Report framing",
        'edge_deploy':  "Chapter core learning objective",
    }

    _rows = ""
    for _key, _label in _option_labels.items():
        _is_selected = locals()[_key].value
        _is_correct  = _key in _correct_set
        if _is_selected and _is_correct:
            _icon, _bg, _col = "✅", "#f0fdf4", "#15803d"
        elif _is_selected and not _is_correct:
            _icon, _bg, _col = "❌", "#fef2f2", "#dc2626"
        elif not _is_selected and _is_correct:
            _icon, _bg, _col = "◉", "#fffbeb", "#d97706"
        else:
            _icon, _bg, _col = "○", "#f8fafc", "#94a3b8"
        _rows += f"""
        <div style="background:{_bg}; border-radius:8px; padding:10px 14px; margin:4px 0;
                    display:flex; align-items:center; gap:10px;">
            <span style="font-size:1rem;">{_icon}</span>
            <span style="color:{_col}; font-size:0.9rem; font-weight:{'700' if _is_selected or _is_correct else '400'};">
                {_label}
            </span>
        </div>"""

    _explanation = """
    <div style="margin-top:14px; font-size:0.9rem; color:#1e293b; line-height:1.7;">
        <strong>Track-specific:</strong> story, stakeholder, device assumptions,
        metrics, guardrails, and report framing should all change with the track. <br/><br/>
        <strong>Shared across tracks:</strong> the chapter's core learning objective
        stays the same. Everyone learns the same MLSys idea, but each track
        materializes it in a different setting.
    </div>
    """

    _title = "✅ Exactly right." if _exactly_right else (
        "⚠️ Partially right — review the highlighted options." if not _has_wrong else
        "⚠️ Not quite — the core lesson should stay shared across tracks."
    )
    _border = "#16a34a" if _exactly_right else ("#f59e0b" if not _has_wrong else "#ef4444")
    _bg_outer = "#f0fdf4" if _exactly_right else ("#fffbeb" if not _has_wrong else "#fef2f2")

    # Keep feedback in the same answer box so students know where a click
    # produced its explanation.
    _items = [
        mo.md("""**Part B prediction.** When you choose a track, which parts of
    later labs should change automatically?"""),
        mo.md("""Select **all** that should change because of the track:"""),
        model_size,
        quantization,
        move_server,
        faster_gpu,
        edge_deploy,
        mo.Html(f"""
        <div style="background:{_bg_outer}; border:1.5px solid {_border};
                    border-radius:10px; padding:18px 20px; margin-top:8px;">
            <div style="font-weight:700; font-size:0.95rem; color:{_border}; margin-bottom:10px;">{_title}</div>
            {_rows}
            {_explanation}
        </div>
        """),
    ]

    if not _exactly_right:
        _items.append(mo.callout(mo.md(
            "**Track rule of thumb:** the track should change the narrative and "
            "constraints, but it should not change what concept the lab teaches."
        ), kind="warn"))

    _items.append(mo.accordion({
        "Why keep one shared concept?": mo.md("""
    If every track taught a different idea, students could not compare notes.
    The track should change the concrete material: device, workload, metric,
    guardrail, stakeholder, and report. The underlying concept stays common.
    """)
    }))

    mo.vstack(_items)
    return

# ─── CONCEPT 3: THE DEPLOYMENT REGIMES ────────────────────────────────────────

@app.cell
def _(CANONICAL_TRACKS, check1, check2empty, mo, track_display_label, v1_00_track_story):
    mo.stop(check1.value is None or check2empty())

    _color_by_track = {
        "iphone": "#f59e0b",
        "oura_ring": "#10b981",
        "robotaxi": "#ef4444",
        "cloud_fleet": "#6366f1",
    }
    _cards = ""
    for _profile in CANONICAL_TRACKS:
        _color = _color_by_track.get(_profile.track_id, "#6366f1")
        _story = v1_00_track_story(_profile.track_id)
        _constraints = ", ".join(_profile.dominant_constraints)
        _metrics = ", ".join(_profile.primary_metrics)
        _guardrails = ", ".join(_profile.guardrail_metrics)
        _cards += f"""
            <div style="background: white; border: 1px solid {_color}44; border-radius: 12px; padding: 20px;">
                <div style="font-weight: 800; color: #1e293b; font-size: 1.0rem; margin-bottom: 4px;">
                    {track_display_label(_profile)}
                </div>
                <div style="font-size: 0.78rem; color: {_color}; font-weight: 700; margin-bottom: 10px;">
                    {_profile.category}
                </div>
                <div style="color: #475569; font-size: 0.87rem; line-height: 1.6; margin-bottom: 12px;">
                    {_profile.narrative}
                </div>
                <div style="background: {_color}12; border-radius: 8px; padding: 8px 12px;
                            font-size: 0.78rem; color: #334155; line-height: 1.55;">
                    <strong>Your role:</strong> {_profile.stakeholder}<br/>
                    <strong>Stakeholder asks:</strong> {_story['stakeholder_question']}<br/>
                    <strong>Primary metrics:</strong> {_metrics}<br/>
                    <strong>Guardrails:</strong> {_guardrails}<br/>
                    <strong>Dominant constraints:</strong> {_constraints}<br/>
                    <strong>Likely failure:</strong> {_story['likely_failure']}<br/>
                    <strong>Report frame:</strong> {_story['report_frame']}
                </div>
            </div>
        """

    mo.vstack([
        mo.md("---"),
        mo.md("""
        ## The Four Canonical Tracks

        The course supports four canonical tracks. You will choose one as your
        recurring operating envelope. Later labs will change the story, stakeholder,
        constraints, metrics, failure mode, and report frame from that choice while
        keeping the core concept shared.
        """),
        mo.Html(f"""
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; margin: 16px 0;">
            {_cards}
        </div>
        """),
        mo.callout(
            mo.md(
                "**Track choice is a learning lens.** You are not choosing a career path. "
                "You are choosing which deployment constraints you will keep returning to "
                "as each new ML systems idea is introduced."
            ),
            kind="info",
        ),
    ])
    return

# ─── CHECK 3 (lab workflow) ───────────────────────────────────────────────────

@app.cell
def _(check1, check2empty, mo):
    mo.stop(check1.value is None or check2empty())

    check3 = mo.ui.radio(
        options={
            "A)  Read the case, predict first, manipulate controls, inspect evidence, decide, then report": "A",
            "B)  Skip the case and tune controls until the answer looks right": "B",
            "C)  Download a report before making the required decisions": "C",
            "D)  Treat the track as a separate topic from the lab": "D",
        },
        label="""**Part C prediction.** What workflow should you expect to repeat in later labs?""",
    )
    return (check3,)

@app.cell
def _(check1, check2empty, check3, mo):
    mo.stop(check1.value is None or check2empty() or check3.value is not None)

    mo.vstack([
        check3,
    ])
    return

@app.cell
def _(check1, check2empty, check3, mo):
    mo.stop(check1.value is None or check2empty() or check3.value is None)

    _correct = check3.value == "A"
    _feedback = {
        "A": (
            "**Correct.** Later labs repeat this order so the interface stays familiar "
            "while the technical ideas become more substantial."
        ),
        "B": (
            "**Not quite.** Controls are there to create evidence, not to hide the case. "
            "You should understand the situation before changing settings."
        ),
        "C": (
            "**Not quite.** The report should come after the required parts, summary, "
            "and takeaways. It is the artifact of your work, not the starting point."
        ),
        "D": (
            "**Not quite.** The track should be threaded through the lab. It changes "
            "the scenario, evidence, and report framing for the same shared concept."
        ),
    }

    mo.vstack([
        check3,
        mo.callout(
            mo.md(_feedback[check3.value]),
            kind="success" if _correct else "warn",
        ),
    ])
    return

# ═══════════════════════════════════════════════════════════════════════════════
# ZONE C: INTERFACE ORIENTATION
# ═══════════════════════════════════════════════════════════════════════════════

# _act_why: "Before Lab 01, students should recognize the recurring case -> prediction -> manipulation -> evidence -> decision rhythm without advanced terminology."

# ─── INTERFACE ORIENTATION INTRO ───────────────────────────────────────────────

@app.cell
def _(check1, check2empty, check3, mo):
    mo.stop(check1.value is None or check2empty() or check3.value is None)

    mo.vstack([
        mo.md("---"),
        mo.md("""
        ## Part C - The Repeated Lab Ritual

        **Scenario.** A later lab opens with a stakeholder case and a system that
        is not yet defensible. You do not start by tuning randomly. You first make
        a structured prediction, then use the controls to create evidence.

        Starting from Lab 01, every lab follows the same rhythm: read a case,
        make a prediction, manipulate a small number of controls, inspect the
        evidence, choose a decision, and write the report you would defend.

        Before you begin Lab 01, spend two minutes with the tour below. The goal
        is only to recognize the pattern. The technical vocabulary arrives later,
        one lab at a time.
        """),
    ])
    return

@app.cell
def _(check1, check2empty, check3, mo):
    mo.stop(check1.value is None or check2empty() or check3.value is None)

    v1_00_evidence_slider = mo.ui.slider(
        start=0,
        stop=100,
        step=5,
        value=55,
        label="Evidence strength:",
    )
    return (v1_00_evidence_slider,)

@app.cell
def _(
    COLORS, check2empty, mo, check1,
    check3, v1_00_evidence_slider,
):
    mo.stop(check1.value is None or check2empty() or check3.value is None)

    # ── SIMPLE LAB RHYTHM DIAGRAM ────────────────────────────────────
    _zone_html = """
    <div style="display:grid; grid-template-columns:1fr 1fr; gap:14px; margin:16px 0;">

        <div style="background:#f0f4ff; border:1.5px solid #c7d2fe; border-radius:10px;
                    padding:16px; border-top:4px solid #6366f1;">
            <div style="font-weight:800; color:#3730a3; font-size:0.9rem; margin-bottom:6px;">
                1 · Read The Case
            </div>
            <div style="color:#4338ca; font-size:0.83rem; line-height:1.55;">
                Each lab starts with a short situation. Your selected track
                changes the stakeholder, the device, and the constraint you
                should care about first.
            </div>
        </div>

        <div style="background:#f0fdf4; border:1.5px solid #bbf7d0; border-radius:10px;
                    padding:16px; border-top:4px solid #16a34a;">
            <div style="font-weight:800; color:#14532d; font-size:0.9rem; margin-bottom:6px;">
                2 · Make A Prediction
            </div>
            <div style="color:#166534; font-size:0.83rem; line-height:1.55;">
                Before looking at the result, you record what you think will
                happen. A prediction is not graded as a trick question; it gives you
                something concrete to compare against the evidence.
            </div>
        </div>

        <div style="background:#fff7ed; border:1.5px solid #fed7aa; border-radius:10px;
                    padding:16px; border-top:4px solid #ea580c;">
            <div style="font-weight:800; color:#9a3412; font-size:0.9rem; margin-bottom:6px;">
                3 · Explore Evidence
            </div>
            <div style="color:#7c2d12; font-size:0.83rem; line-height:1.55;">
                Sliders, choices, tables, and charts show what changes when you
                adjust the system. These controls are the pieces that update
                the evidence.
            </div>
        </div>

        <div style="background:#fffbeb; border:1.5px solid #fde68a; border-radius:10px;
                    padding:16px; border-top:4px solid #d97706;">
            <div style="font-weight:800; color:#92400e; font-size:0.9rem; margin-bottom:6px;">
                4 · Decide And Report
            </div>
            <div style="color:#78350f; font-size:0.83rem; line-height:1.55;">
                After reading the evidence, you choose a defensible answer and
                write a short rationale. The downloaded report captures that
                reasoning for review.
            </div>
        </div>

    </div>
    """

    _evidence_value = v1_00_evidence_slider.value
    if _evidence_value < 35:
        _evidence_label = "weak"
        _constraint_status = "unproven"
        _decision_status = "do not report yet"
        _status_color = COLORS["RedLine"]
    elif _evidence_value < 70:
        _evidence_label = "medium"
        _constraint_status = "partly checked"
        _decision_status = "needs rationale"
        _status_color = COLORS["OrangeLine"]
    else:
        _evidence_label = "strong"
        _constraint_status = "ready to defend"
        _decision_status = "report-ready"
        _status_color = COLORS["GreenLine"]

    # ── LIVE COMPONENT TOUR via mo.ui.tabs ────────────────────────────
    _tab_overview = mo.vstack([
        mo.md("""
        **Lab rhythm** — later labs repeat the same learning loop.

        The technical topic changes from lab to lab, but the student move stays
        familiar: read the case, make a prediction, move the controls, inspect the
        evidence, decide, and explain.
        """),
        mo.Html(_zone_html),
        mo.callout(
            mo.md("The point of Lab 00 is orientation. You should leave knowing what role your track plays and how a later lab page asks you to work."),
            kind="info"
        ),
    ])

    _tab_levers = mo.vstack([
        mo.md("""
        **Controls change evidence.** A radio button that asks for your prediction only
        records your expectation. Sliders and dropdowns are the controls that
        change plots, tables, and status cards.
        """),
        mo.hstack([
            mo.vstack([
                mo.md("**Example control**"),
                v1_00_evidence_slider,
            ], gap=1),
            mo.Html(f"""
            <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px;
                        padding:16px; min-width:220px;">
                <div style="font-size:0.7rem; font-weight:700; color:#94a3b8;
                            text-transform:uppercase; margin-bottom:8px;">Evidence Preview</div>
                <div style="font-size:0.82rem; color:#475569; line-height:1.8;">
                    Current evidence: <strong style="color:{_status_color}">{_evidence_label}</strong><br/>
                    Track constraint: <strong style="color:{_status_color}">{_constraint_status}</strong><br/>
                    Decision status: <strong style="color:{_status_color}">{_decision_status}</strong>
                </div>
                <div style="margin-top:10px; font-size:0.72rem; color:#94a3b8; font-style:italic;">
                    In later labs, real plots and tables update as you move controls.
                    Here the slider only previews how evidence gates decisions.
                </div>
            </div>
            """),
        ], gap=2, justify="start"),
        mo.callout(
            mo.md("Use predictions to learn from surprise. Use controls to create the evidence you will reason from."),
            kind="warn",
        ),
    ])

    _tab_prediction = mo.vstack([
        mo.md("""
        **Predict first, then test.**

        Many labs ask for an initial answer before you move a control. That
        answer is a learning checkpoint. It should not make the chart jump.
        """),
        mo.Html("""
        <div style="background:#f8fafc; border-radius:10px; padding:20px; border-left:4px solid #16a34a; border:1px solid #dbeafe;">
            <div style="font-size:0.72rem; font-weight:700; color:#6366f1;
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:10px;">
                Step 1 - Before Evidence
            </div>
            <div style="color:#1e293b; font-size:0.9rem; line-height:1.6; margin-bottom:14px;">
                <strong>Scenario:</strong> The system is missing its target.<br/><br/>
                <strong>Prediction:</strong> What do you think is the most likely reason?
            </div>
            <div style="display:flex; gap:12px; flex-wrap:wrap;">
                <div style="background:#ecfdf5; border:1px solid #86efac;
                            border-radius:8px; padding:8px 16px; color:#166534; font-size:0.85rem;
                            font-weight:600; cursor:pointer;">
                    A) The data
                </div>
                <div style="background:#ecfdf5; border:1px solid #86efac;
                            border-radius:8px; padding:8px 16px; color:#166534; font-size:0.85rem;
                            font-weight:600; cursor:pointer;">
                    B) The model
                </div>
                <div style="background:#ecfdf5; border:1px solid #86efac;
                            border-radius:8px; padding:8px 16px; color:#166534; font-size:0.85rem;
                            font-weight:600; cursor:pointer;">
                    C) The device
                </div>
            </div>
            <div style="margin-top:12px; font-size:0.78rem; color:#64748b;">
                This records your starting point. The evidence controls come next.
            </div>
        </div>
        """),
        mo.md("""
        Being wrong is useful here. The lab is asking you to notice how evidence
        changes your mind, not to already know the answer.
        """),
    ])

    _tab_mathpeek = mo.vstack([
        mo.md("""
        **Optional details** keep the main page readable.

        Later labs sometimes include a formula, a definition, or a short reference.
        Those details live behind expandable sections so you can open them when
        you need them and keep reading when you do not.
        """),
        mo.accordion({
            "Open a sample detail": mo.md("""
            A later lab might explain a metric in one paragraph, show the
            equation behind it, or name the chapter section that introduced it.
            The main task still stays visible above the detail.
            """),
        }),
        mo.callout(
            mo.md("You do not need every detail at once. Open details when they help you answer the current question."),
            kind="info",
        ),
    ])

    _tour_tabs = mo.ui.tabs({
        "Lab Rhythm":       _tab_overview,
        "Controls":         _tab_levers,
        "Predict First":    _tab_prediction,
        "Optional Details": _tab_mathpeek,
    })

    mo.vstack([
        _tour_tabs,
        mo.Html("""
        <div style="background:#0f172a; border-radius:10px; padding:16px 22px; margin-top:16px;
                    border:1px solid #1e293b; display:flex; align-items:center; gap:16px;">
            <div style="font-size:1.3rem;">✅</div>
            <div style="font-size:0.87rem; color:#94a3b8; line-height:1.6;">
                <strong style="color:#e2e8f0;">Interface orientation complete.</strong>
                You now know the repeated loop: case, prediction, manipulation, evidence,
                decision, and report. Later labs add technical ideas gradually
                inside that same structure.
            </div>
        </div>
        """),
    ])
    return

# ═══════════════════════════════════════════════════════════════════════════════
# ZONE D: CLOSING
# ═══════════════════════════════════════════════════════════════════════════════

# ─── DEPLOYMENT CONTEXT SELECTION ─────────────────────────────────────────────

@app.cell
def _(check1, check2empty, check3, mo):
    mo.stop(
        check1.value is None or check2empty() or check3.value is None,
        mo.md("_Complete the orientation checks above to unlock your track selection._")
    )

    mo.vstack([
        mo.md("---"),
        mo.md("""
        ## Part D - Choose An Operating Envelope For The Ledger

        You have now seen what a track does. For the rest of Volume I, you will
        carry one deployment context as your primary learning lens. Every student
        learns the same concepts; the track changes the concrete story and evidence.

        **Scenario.** Future labs need to know which stakeholder and operating
        envelope to load before they can choose defaults. This is why Lab 00 saves
        the track in the local Design Ledger instead of treating it as a temporary
        page setting.

        **This is not a career choice.** It is a way to keep the examples coherent
        from lab to lab. Later labs will read this choice from the local Design
        Ledger and adapt narrative, device assumptions, metrics, guardrails,
        failure modes, and report framing automatically.
        """),
    ])
    return

@app.cell
def _(check1, check2empty, check3, mo):
    mo.stop(check1.value is None or check2empty() or check3.value is None)

    context_selector = mo.ui.radio(
        options={
            "📱  iPhone — mobile privacy, thermal, battery, and latency": "iphone",
            "💍  Oura Ring — wearable SRAM, flash, OTA, and battery": "oura_ring",
            "🚕  RoboTaxi — edge p99 latency, reliability, and safety margin": "robotaxi",
            "☁️  Cloud Fleet — throughput, p99 SLA, utilization, cost, and carbon": "cloud_fleet",
        },
        label="Select the track you will focus on throughout this curriculum:",
    )
    return (context_selector,)

@app.cell
def _(check1, check2empty, check3, context_selector, mo):
    mo.stop(
        check1.value is None
        or check2empty()
        or check3.value is None
        or context_selector.value is not None
    )

    _context_prompt = mo.vstack([
        context_selector,
        mo.md("_Select your deployment context above._"),
    ], align="center")
    _context_prompt
    return

# ─── CONTEXT REVEAL + STAKEHOLDER MESSAGE + LEDGER INIT ───────────────────────

@app.cell(hide_code=True)
def _(DecisionLog):
    decision_input, decision_ui = DecisionLog(
        placeholder="I chose this operating envelope because its recurring constraint will force me to reason about..."
    )
    return decision_input, decision_ui

@app.cell
def _(
    COLORS,
    check1,
    check2empty,
    check2value_list,
    check3,
    context_selector,
    decision_input,
    decision_ui,
    get_track_profile,
    lab_metadata,
    ledger,
    mo,
    track_display_label,
    track_context,
    track_arc_context,
    v1_00_track_story,
):
    mo.stop(
        check1.value is None
        or check2empty()
        or check3.value is None
        or context_selector.value is None
    )

    _track_id = context_selector.value
    _track_profile = get_track_profile(_track_id)
    _story = v1_00_track_story(_track_id)
    _contexts = {
        "cloud_fleet": {
            "color":     COLORS["BlueLine"],
            "bg":        COLORS["BlueL"],
            "persona":   "Your CTO",
            "quote": (
                "The model is not the whole service. Show me throughput, p99, cost, "
                "utilization, and carbon before you call this deployable."
            ),
            "arc": [
                ("Labs 01–04", "Foundations",
                 "Learn how to diagnose whether the first fix belongs to data, model design, or infrastructure"),
                ("Labs 05–08", "Build",
                 "Build and measure the pieces of a serving stack instead of treating the model as a black box"),
                ("Labs 09–11", "Optimize",
                 "Reduce cost and latency while keeping quality and reliability visible"),
                ("Labs 12–14", "Deploy",
                 "Benchmark, monitor, and operate a production serving system at scale"),
            ],
        },
        "robotaxi": {
            "color":     COLORS["RedLine"],
            "bg":        COLORS["RedL"],
            "persona":   "Your Safety Director",
            "quote": (
                "Average latency is not a safety case. Bring me p99 and p999 evidence, "
                "rare-event replay, and a fallback plan."
            ),
            "arc": [
                ("Labs 01–04", "Foundations",
                 "Learn how safety-critical systems turn one failure symptom into evidence-backed diagnosis"),
                ("Labs 05–08", "Build",
                 "Build the parts of an inference path and measure what happens under realistic load"),
                ("Labs 09–11", "Optimize",
                 "Simplify and accelerate models while protecting worst-case behavior"),
                ("Labs 12–14", "Deploy",
                 "Validate the system with safety, fallback, and monitoring evidence"),
            ],
        },
        "iphone": {
            "color":     COLORS["OrangeLine"],
            "bg":        COLORS["OrangeL"],
            "persona":   "Your UX Director",
            "quote": (
                "A local model is only useful if the phone still feels responsive, "
                "private, and comfortable after sustained use."
            ),
            "arc": [
                ("Labs 01–04", "Foundations",
                 "Learn how local-device constraints change the first fix you should try"),
                ("Labs 05–08", "Build",
                 "Build efficient model components and connect them to user-visible responsiveness"),
                ("Labs 09–11", "Optimize",
                 "Compress and tune models while protecting quality, battery, and heat"),
                ("Labs 12–14", "Deploy",
                 "Benchmark sustained behavior under realistic mobile workloads"),
            ],
        },
        "oura_ring": {
            "color":     COLORS["GreenLine"],
            "bg":        COLORS["GreenL"],
            "persona":   "Your Hardware Lead",
            "quote": (
                "Every byte and every radio wakeup has an owner. If the firmware, "
                "model, sensor window, and OTA package do not fit together, it does not ship."
            ),
            "arc": [
                ("Labs 01–04", "Foundations",
                 "Learn how tiny-device limits force early choices about data windows, model size, and memory"),
                ("Labs 05–08", "Build",
                 "Build compact model pieces and see how memory layout affects whether they fit"),
                ("Labs 09–11", "Optimize",
                 "Shrink the model while protecting accuracy, update size, and battery use"),
                ("Labs 12–14", "Deploy",
                 "Validate that the full sensing and inference path fits the tiny hardware envelope"),
            ],
        },
    }

    _t = _contexts[_track_id]
    _rationale = decision_input.value or ""

    # Persist to Design Ledger
    ledger.save(track=_track_id, chapter=0, design={
        "deployment_context": _track_id,
        "track_id": _track_profile.track_id,
        "track_label": _track_profile.label,
        "track_category": _track_profile.category,
        "hardware_ref": _track_profile.hardware_ref,
        "system_ref": _track_profile.system_ref,
        "primary_metrics": _track_profile.primary_metrics,
        "guardrail_metrics": _track_profile.guardrail_metrics,
        "dominant_constraints": _track_profile.dominant_constraints,
        "orientation_goal_answer": check1.value,
        "orientation_goal_correct": check1.value == "C",
        "track_change_selections": check2value_list(),
        "lab_workflow_answer": check3.value,
        "lab_workflow_correct": check3.value == "A",
        "chapter_lab_invariant": "MLSys labs teach deployed behavior under operating envelopes.",
        "stakeholder": _track_profile.stakeholder,
        "stakeholder_question": _story["stakeholder_question"],
        "first_bottleneck_prediction": _story["first_bottleneck"],
        "likely_failure_mode": _story["likely_failure"],
        "report_frame": _story["report_frame"],
        "track_rationale": _rationale,
    })

    _arc_rows = "".join([
        f"""<tr>
            <td style="padding:9px 14px; font-size:0.8rem; color:#64748b; font-weight:600;
                       white-space:nowrap; border-bottom:1px solid #f1f5f9;">{phase}</td>
            <td style="padding:9px 14px; font-size:0.82rem; font-weight:700;
                       color:{_t['color']}; white-space:nowrap;
                       border-bottom:1px solid #f1f5f9;">{label}</td>
            <td style="padding:9px 14px; font-size:0.82rem; color:#475569;
                       border-bottom:1px solid #f1f5f9;">{desc}</td>
        </tr>"""
        for phase, label, desc in _t["arc"]
    ])

    mo.vstack([
        mo.vstack([context_selector], align="center"),
        mo.md("---"),
        decision_ui,
        track_context(_track_profile),
        track_arc_context(_track_profile, lab_metadata.lab_id),

        # Stakeholder message
        mo.Html(f"""
        <div style="border-left:4px solid {_t['color']}; background:{_t['bg']};
                    border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
            <div style="font-size:0.72rem; font-weight:700; color:{_t['color']};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                Incoming Message · {_t['persona']}
            </div>
            <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                "{_t['quote']}"
            </div>
        </div>
        """),

        # Mission card
        mo.Html(f"""
        <div style="border:2px solid {_t['color']}20; border-radius:12px; padding:24px;
                    background:white; margin:12px 0; box-shadow:0 2px 8px rgba(0,0,0,0.04);">
            <div style="font-size:0.72rem; font-weight:700; text-transform:uppercase;
                        letter-spacing:0.1em; color:{_t['color']}; margin-bottom:6px;">
                🎖️ Deployment Context Confirmed
            </div>
            <div style="font-size:1.25rem; font-weight:800; color:#0f172a; margin-bottom:4px;">
                {track_display_label(_track_profile)} · {_track_profile.stakeholder}
            </div>
            <div style="font-size:0.88rem; color:#475569; margin-bottom:4px; line-height:1.5;">
                <strong>North Star:</strong> {_track_profile.narrative}
            </div>
            <div style="font-size:0.88rem; margin-bottom:18px; line-height:1.5;">
                <strong style="color:{_t['color']};">Arch Nemesis:</strong>
                <span style="color:#334155;"> {", ".join(_track_profile.dominant_constraints)}</span>
            </div>
            <div style="background:{_t['bg']}; border-radius:10px; padding:12px 14px;
                        font-size:0.84rem; color:#334155; line-height:1.6; margin-bottom:18px;">
                <strong>First bottleneck to watch:</strong> {_story['first_bottleneck']}<br/>
                <strong>Likely failure:</strong> {_story['likely_failure']}<br/>
                <strong>Report frame:</strong> {_story['report_frame']}
            </div>
            <div style="font-size:0.75rem; font-weight:700; text-transform:uppercase;
                        letter-spacing:0.07em; color:#94a3b8; margin-bottom:8px;">
                Your 14-Lab Arc
            </div>
            <table style="width:100%; border-collapse:collapse;">
                <tbody>{_arc_rows}</tbody>
            </table>
        </div>
        """),

        mo.callout(
            mo.md(
                f"**Design Ledger initialized** — track: {track_display_label(_track_profile)}. "
                "Your track pre-loads hardware defaults and scenario constraints "
                "in every lab from Lab 01 onward. Proceed to **Lab 01: ML Introduction**."
            ),
            kind="success",
        ),
    ])
    return

# ─── CELL 20: SYNTHESIS ────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(COLORS, check1, check2empty, check3, context_selector, mo):
    mo.stop(
        check1.value is None
        or check2empty()
        or check3.value is None
        or context_selector.value is None
    )

    mo.vstack([
        mo.md("---"),

        # ── KEY TAKEAWAYS ──
        mo.Html(f"""
        <div style="background: {COLORS['Surface2']}; border: 1px solid {COLORS['Border']};
                    border-radius: 12px; padding: 24px 28px; margin: 16px 0;">
            <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                        text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 12px;">
                Big Takeaways
            </div>
            <div style="font-size: 0.92rem; color: {COLORS['Text']}; line-height: 1.75;">
                <div style="margin-bottom: 10px;">
                    <strong>1. These labs practice deployed-behavior reasoning.</strong>
                    You will read a case, make a prediction, manipulate controls,
                    inspect evidence, choose a defensible decision, and download a short report.
                </div>
                <div style="margin-bottom: 10px;">
                    <strong>2. A track is an operating envelope.</strong>
                    iPhone, Oura Ring, RoboTaxi, and Cloud Fleet change the story,
                    stakeholder, device assumptions, constraints, metrics, failure modes,
                    guardrails, and report framing.
                </div>
                <div>
                    <strong>3. The ledger makes decisions cumulative.</strong>
                    The selected track becomes the default context future labs use when
                    they ask which amount, budget, or failure mode is binding.
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
                    <strong>Lab 01: The AI Triad.</strong> This lab established your track
                    and the repeated lab rhythm. Lab 01 asks your first diagnosis
                    question: when a system is failing, should the first fix focus on
                    the data, the algorithm, or the machine?
                </div>
            </div>

            <!-- Textbook & TinyTorch -->
            <div style="flex: 1; min-width: 280px; background: white;
                        border: 1px solid {COLORS['Border']}; border-radius: 12px;
                        padding: 20px 24px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['GreenLine']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 8px;">
                    Textbook &amp; TinyTorch
                </div>
                <div style="font-size: 0.88rem; color: {COLORS['TextSec']}; line-height: 1.6;">
                    <strong>Read:</strong> Chapter 1: Introduction for why production ML is a
                    system, not just a model in a notebook.<br/>
                    <strong>Build:</strong> TinyTorch starts in Module 01 with the foundations
                    you will keep using as the labs become more technical.
                </div>
            </div>

        </div>
        """),

        mo.accordion({
            "Self-Assessment: Can you answer these?": mo.md("""
    1. Which track (☁️ Cloud Fleet, 📱 iPhone, 🚕 RoboTaxi, or 💍 Oura Ring) did you choose, and what constraint will that track keep bringing back?

    2. Why is a model that works in a notebook not automatically a deployed ML system?

    3. What is the recurring order of work in later labs: what case do you read, what prediction do you make, what controls do you manipulate, what evidence do you inspect, what decision do you defend, and what report do you produce?

    4. How will your selected track change the meaning of later quantities such as latency, energy, memory, cost, utilization, or carbon?

    *If you cannot answer all four from memory, revisit the track cards and interface orientation.*
    """)
        }),
    ])
    return

@app.cell
def _(
    build_lab_report,
    check1,
    check2empty,
    check2value_list,
    check3,
    context_selector,
    decision_input,
    get_track_profile,
    lab_big_takeaways,
    lab_learning_objectives,
    lab_metadata,
    mo,
    report_export_panel,
    v1_00_track_story,
):
    mo.stop(
        check1.value is None
        or check2empty()
        or check3.value is None
        or context_selector.value is None,
        mo.md("_Complete the orientation checks and choose your track to unlock the local report._"),
    )

    _track_id = context_selector.value
    _profile = get_track_profile(_track_id)
    _story = v1_00_track_story(_track_id)
    _selected = f"{_profile.label} ({_profile.category})"
    _report = build_lab_report(
        lab_metadata,
        track=_profile.label,
        scenario="Lab 00 track selection and lab-interface orientation",
        learning_objectives=lab_learning_objectives,
        predictions={
            "orientation_goal": check1.value,
            "track_specific_elements": ", ".join(check2value_list()),
            "recurring_lab_workflow": check3.value,
            "track_implied_first_bottleneck": _story["first_bottleneck"],
        },
        evidence_summary={
            "selected_track": _selected,
            "hardware_ref": _profile.hardware_ref,
            "system_ref": _profile.system_ref or "single-device profile",
            "dominant_constraints": ", ".join(_profile.dominant_constraints),
            "likely_failure_mode": _story["likely_failure"],
            "report_frame": _story["report_frame"],
            "amount_reasoning": _story["amount_reasoning"],
        },
        final_decision=(
            f"Use {_profile.label} as the student's canonical track for subsequent labs, "
            f"watching {_story['first_bottleneck']} as the first recurring bottleneck."
        ),
        big_takeaways=lab_big_takeaways,
        reflections={
            "orientation": "Later labs use the same case, prediction, manipulation, evidence, decision, and report rhythm.",
            "track_choice": f"{_profile.label} changes the scenario and constraints while preserving the shared concept.",
            "track_rationale": decision_input.value or "No written rationale entered.",
            "report_readiness": "The report unlocks only after the orientation checks, synthesis, and track choice are complete.",
        },
        residual_risk=(
            "Lab 00 records the student's track and interface orientation. Later labs "
            "must still build evidence for each concrete design decision."
        ),
        source_trace={
            "track_id": _profile.track_id,
            "hardware_ref": _profile.hardware_ref,
            "system_ref": _profile.system_ref or "single-device profile",
            "source_policy": _profile.source_policy,
            "report_builder": "mlsysbook_labs.build_lab_report",
            "orientation_story_source": "labs/vol1/lab_00_introduction.py::v1_00_track_story",
        },
        result_snapshot={
            "track_id": _profile.track_id,
            "track_label": _profile.label,
            "hardware_ref": _profile.hardware_ref,
            "system_ref": _profile.system_ref,
            "orientation_goal_correct": check1.value == "C",
            "lab_workflow_correct": check3.value == "A",
            "first_bottleneck_prediction": _story["first_bottleneck"],
            "likely_failure_mode": _story["likely_failure"],
            "report_frame": _story["report_frame"],
        },
    )

    mo.vstack([
        mo.md("## Download Report"),
        mo.callout(
            mo.md(
                "Your Lab 00 orientation report is generated locally from the current "
                "track choice and completed orientation checks."
            ),
            kind="info",
        ),
        report_export_panel(_report),
    ])
    return

# ─── CELL 21: LEDGER_HUD ───────────────────────────────────────────────────────
@app.cell
def _(COLORS, context_selector, get_track_profile, mo, track_display_label):
    mo.stop(context_selector.value is None)

    _track = context_selector.value
    _color_map = {
        "cloud":  COLORS["BlueLine"],
        "edge":   COLORS["RedLine"],
        "mobile": COLORS["OrangeLine"],
        "tiny":   COLORS["GreenLine"],
        "cloud_fleet": COLORS["BlueLine"],
        "robotaxi": COLORS["RedLine"],
        "iphone": COLORS["OrangeLine"],
        "oura_ring": COLORS["GreenLine"],
        "NONE":   "#475569",
    }
    _hud_color  = _color_map.get(_track, "#475569")
    _hud_status = "Active — Chapter 0"
    _hud_track = track_display_label(get_track_profile(_track))

    mo.Html(f"""
    <div style="display:flex; gap:28px; align-items:center; padding:12px 24px;
                background:#0f172a; border-radius:10px; margin-top:32px;
                font-family:'SF Mono','Fira Code',monospace; font-size:0.8rem;
                border:1px solid #1e293b;">
        <div style="color:#475569; font-weight:600; letter-spacing:0.06em;">🗂️ DESIGN LEDGER</div>
        <div>
            <span style="color:#475569;">Context: </span>
            <span style="color:{_hud_color}; font-weight:700;">{_hud_track}</span>
        </div>
        <div>
            <span style="color:#475569;">Chapter: </span>
            <span style="color:#e2e8f0;">0</span>
        </div>
        <div>
            <span style="color:#475569;">Status: </span>
            <span style="color:{'#4ade80' if _track != 'NONE' else '#f87171'};">{_hud_status}</span>
        </div>
    </div>
    """)
    return

# --- Auxiliary methods ---------------------------------------------------------
@app.cell
def _(edge_deploy, faster_gpu, model_size, move_server, quantization):
    _check2_labels = {
        'model_size': 'story and stakeholder voice',
        'quantization': 'device and hardware assumptions',
        'move_server': 'primary metric and guardrails',
        'faster_gpu': 'report framing',
        'edge_deploy': "chapter core learning objective",
    }

    def check2empty():
        return not (model_size.value or quantization.value or move_server.value or faster_gpu.value or edge_deploy.value)

    def check2value_list():
        check2values = []
        # have to use items directly here for dependency evaluation
        if (model_size.value or quantization.value or move_server.value or faster_gpu.value or edge_deploy.value):
            for item in ['model_size', 'quantization', 'move_server' ,'faster_gpu', 'edge_deploy']:
                if globals()[item].value:
                    check2values.append(_check2_labels[item])
        return check2values

    return check2empty, check2value_list

if __name__ == "__main__":
    app.run()

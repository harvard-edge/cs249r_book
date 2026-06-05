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
# Four sections:
#   1. Concept blocks with structured checks (3 total)
#   2. Interface Orientation — case, guess, controls, evidence, decision, report
# No physics instruments (introduced in Lab 01+).
# No prediction locks in anger (students haven't read Chapter 1 yet).
# Progressive disclosure: each check gates the next concept.
#
# Concepts covered (all from pre-reading context, no chapter required):
#   1. The 95% Problem — ML systems ≠ ML models
#   2. Physical constraints partition deployment into distinct regimes
#   3. Constraints are immovable — the choice of regime is the architecture
#   4. UI scaffolding — every recurring component demonstrated before Lab 01
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
        "Identify why production ML systems are constrained by infrastructure, not only model accuracy.",
        "Choose a canonical track and explain the physical constraint that makes it distinct.",
        "Recognize the recurring lab rhythm: read the case, make a guess, explore evidence, decide, and report.",
    )
    lab_big_takeaways = (
        "The same model idea becomes a different systems problem on iPhone, Oura Ring, RoboTaxi, and Cloud Fleet.",
        "Track selection changes hardware facts, constraints, metrics, stakeholder pressure, and report framing.",
        "Later labs build on this track choice, but each lab will introduce only the new concept it needs.",
    )
    return lab_big_takeaways, lab_learning_objectives, lab_metadata

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
                This course is not about machine learning. It is about the infrastructure
                that makes machine learning possible — and the physical laws that govern it.
            </p>
            <div style="display: flex; gap: 12px; flex-wrap: wrap;">
                <span style="background: rgba(99,102,241,0.15); color: #a5b4fc;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(99,102,241,0.25);">
                    Orientation · 3 Concept Checks · Interface Tour
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
                    3 Concept Checks &middot; Interface Tour
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
                <strong>Part A:</strong> 95% infrastructure problem &middot;
                <strong>Part B:</strong> physical regimes &middot;
                <strong>Part C:</strong> recurring lab interface &middot;
                <strong>Synthesis:</strong> choose your track and download the local report.
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
                "If a model reaches 99% accuracy in a Jupyter notebook, what are the 95% of
                engineering problems that still stand between that model and a deployed product
                &mdash; and which physical law determines which problems you cannot solve with software?"
            </div>
        </div>
    </div>
    """)
    return

# ═══════════════════════════════════════════════════════════════════════════════
# ZONE B: CONCEPT CHECKS
# ═══════════════════════════════════════════════════════════════════════════════

# ─── CONCEPT 1: THE 95% PROBLEM ────────────────────────────────────────────────
# _act_why: "You believe ML engineering is about models. The data shows 95% is infrastructure."
@app.cell
def _(mo):
    mo.vstack([
        mo.md("---"),
        mo.md("""
        ## The 95% Problem

        When Google published a study of their internal ML systems in 2015, they found
        something that surprised the field. In a production ML system, the actual model —
        the neural network, the training algorithm, the matrix math — accounts for roughly
        **5% of the total codebase**.

        The other **95%** is infrastructure: data pipelines, serving systems, monitoring,
        hardware resource management, configuration, feature stores, deployment tooling.

        This has a direct implication for how you should think about your role as an engineer:
        """),
        mo.Html("""
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin: 16px 0;">
            <div style="background: #fef2f2; border: 1px solid #fecaca; border-radius: 12px;
                        padding: 20px; border-left: 5px solid #ef4444;">
                <div style="font-weight: 800; color: #991b1b; margin-bottom: 8px;">
                    ML Engineering
                </div>
                <div style="color: #7f1d1d; font-size: 0.9rem; line-height: 1.6;">
                    Build and improve the model. Choose the architecture.
                    Tune hyperparameters. Improve accuracy. <br/><br/>
                    <strong>Optimizes the 5%.</strong>
                </div>
            </div>
            <div style="background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 12px;
                        padding: 20px; border-left: 5px solid #16a34a;">
                <div style="font-weight: 800; color: #14532d; margin-bottom: 8px;">
                    ML Systems Engineering
                </div>
                <div style="color: #14532d; font-size: 0.9rem; line-height: 1.6;">
                    Build the infrastructure that makes the model run reliably
                    at scale, within hardware constraints, in production. <br/><br/>
                    <strong>Optimizes the 95%.</strong>
                </div>
            </div>
        </div>
        """),
        mo.md("""
        A model that achieves 99% accuracy in a Jupyter notebook is **not a product**.
        It becomes a product only when it can run in real-time on real hardware,
        serve thousands of concurrent users, recover from failures, detect when it
        degrades, and update without downtime. That is the engineering this course teaches.
        """),
    ])
    return

# ─── CHECK 1 ───────────────────────────────────────────────────────────────────

@app.cell
def _(mo):
    check1 = mo.ui.radio(
        options={
            "A)  The model architecture — choosing transformers over CNNs": "A",
            "B)  The training algorithm — selecting Adam vs SGD": "B",
            "C)  The serving infrastructure — how the model runs reliably in production": "C",
            "D)  The dataset size — gathering more labeled training examples": "D",
        },
        label="""**Check your understanding.** A startup ships a model with 94% accuracy.
    Six months later, accuracy has silently dropped to 81% in production — but no code
    has changed. As an ML Systems engineer, which part of the system is your *primary*
    domain for diagnosing and fixing this?""",
    )
    return (check1,)

@app.cell
def _(check1, mo):
    mo.stop(
        check1.value is None,
        mo.vstack([
            check1,
            mo.callout(
                mo.md("_Select an answer to continue._"),
                kind="warn",
            ),
        ])
    )

    _correct = check1.value == "C"
    _feedback = {
        "A": (
            "**Not quite.** The architecture hasn't changed — the model itself is unchanged. "
            "The issue is that the *world* changed while the model stayed fixed. "
            "Model architecture is an ML concern; detecting and responding to drift "
            "is a *systems* concern — monitoring, pipelines, retraining triggers."
        ),
        "B": (
            "**Not quite.** The training algorithm only runs during training. "
            "Once the model is deployed, SGD vs Adam no longer matters. "
            "The degradation happened in production — that's the systems layer: "
            "monitoring, data pipelines, serving infrastructure."
        ),
        "C": (
            "**Correct.** The model hasn't changed — but the world it's operating in has. "
            "This is *silent degradation*, one of the defining challenges of ML systems. "
            "Your job is not to debug code; it's to build monitoring that detects when "
            "production data drifts away from training data, and pipelines that respond. "
            "That's the 95%."
        ),
        "D": (
            "**Not quite.** More training data would help if you were retraining — "
            "but the immediate problem is that you don't even *know* the model is degrading "
            "until someone complains. The systems problem is the absence of monitoring. "
            "Data collection is part of the solution, but detecting the problem comes first."
        ),
    }

    mo.vstack([
        check1,
        mo.callout(
            mo.md(_feedback[check1.value]),
            kind="success" if _correct else "warn",
        ),
    ])
    return

# ─── CONCEPT 2: PHYSICAL CONSTRAINTS PARTITION DEPLOYMENT ─────────────────────

@app.cell
def _(check1, mo):
    mo.stop(check1.value is None)

    mo.vstack([
        mo.md("---"),
        mo.md("""
        ## Why Constraints Drive Architecture

        The same model cannot simply be "resized" to run everywhere.
        Three physical laws carve the deployment landscape into distinct regimes
        that no amount of software engineering can bridge:
        """),
        mo.Html("""
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin: 16px 0;">

            <div style="background: white; border: 1px solid #e2e8f0; border-radius: 12px;
                        padding: 18px; border-top: 4px solid #6366f1;">
                <div style="font-size: 1.4rem; margin-bottom: 6px;">⚡</div>
                <div style="font-weight: 800; color: #1e293b; font-size: 0.95rem; margin-bottom: 6px;">
                    The Speed of Light
                </div>
                <div style="color: #64748b; font-size: 0.85rem; line-height: 1.5;">
                    London to New York = ~28 ms one-way, ~56 ms round-trip.
                    A self-driving car that needs a 10 ms decision loop
                    <strong>cannot route to a remote datacenter</strong>.
                    Physics sets this floor. No GPU upgrade helps.
                </div>
            </div>

            <div style="background: white; border: 1px solid #e2e8f0; border-radius: 12px;
                        padding: 18px; border-top: 4px solid #ef4444;">
                <div style="font-size: 1.4rem; margin-bottom: 6px;">🌡️</div>
                <div style="font-weight: 800; color: #1e293b; font-size: 0.95rem; margin-bottom: 6px;">
                    Thermodynamics
                </div>
                <div style="color: #64748b; font-size: 0.85rem; line-height: 1.5;">
                    Heat accumulates faster than a small enclosure can dissipate it.
                    A smartphone running a heavy model continuously
                    <strong>throttles its processor after 90 seconds</strong>.
                    No software fix prevents heat.
                </div>
            </div>

            <div style="background: white; border: 1px solid #e2e8f0; border-radius: 12px;
                        padding: 18px; border-top: 4px solid #10b981;">
                <div style="font-size: 1.4rem; margin-bottom: 6px;">💾</div>
                <div style="font-weight: 800; color: #1e293b; font-size: 0.95rem; margin-bottom: 6px;">
                    Memory Physics
                </div>
                <div style="color: #64748b; font-size: 0.85rem; line-height: 1.5;">
                    Moving data through memory costs energy and takes time.
                    A microcontroller with 256 KB of SRAM
                    <strong>cannot page memory from disk</strong>.
                    If the model doesn't fit, it doesn't run.
                </div>
            </div>

        </div>
        """),
        mo.md("""
        These three constraints — latency floors, power limits, and memory capacity —
        divide the world into four fundamentally different deployment environments.
        Engineers who treat deployment as an afterthought collide with these walls
        after months of architectural work.

        **The insight of ML Systems engineering:** choose your regime *first*,
        because the physics of that regime constrains every design decision that follows.
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
        label="Use a smaller model with fewer parameters"
    )

    quantization = mo.ui.checkbox(
        value=False,
        label="Apply INT8 quantization to reduce precision"
    )

    move_server = mo.ui.checkbox(
        value=False,
        label="Move the datacenter server physically closer"
    )

    faster_gpu = mo.ui.checkbox(
        value=False,
        label="Use a faster GPU with higher TFLOPS"
    )

    edge_deploy = mo.ui.checkbox(
        value=False,
        label="Deploy the model directly on the vehicle"
    )
    return (edge_deploy, faster_gpu, model_size, move_server, quantization)

@app.cell
def _(
    check1,
    edge_deploy,
    faster_gpu,
    mo,
    model_size,
    move_server,
    quantization,
):
    mo.stop(check1.value is None)

    mo.vstack([
        mo.md("""**Check your understanding.** An autonomous vehicle perception system
    is routed to a cloud datacenter 2,000 km away. Round-trip latency is 40 ms.
    The safety requirement is a 10 ms end-to-end decision loop."""),

        mo.md("""Select **all approaches** that could actually solve the latency problem:"""),
        model_size,
        quantization,
        move_server,
        faster_gpu,
        edge_deploy
    ])
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

    _correct_set = {'move_server', 'edge_deploy'}
    _has_wrong     = model_size.value or quantization.value or faster_gpu.value
    _missing_right = not (move_server.value and edge_deploy.value)
    _exactly_right = move_server.value and edge_deploy.value and not _has_wrong

    _option_labels = {
        'model_size':   "Use a smaller model",
        'quantization': "Apply INT8 quantization",
        'move_server':  "Move the server physically closer",
        'faster_gpu':   "Use a faster GPU",
        'edge_deploy':  "Deploy on the vehicle",
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
        <strong>The physics:</strong> The 40 ms latency comes from the speed of light
        across 2,000 km of fiber — approximately 200,000 km/s.
        No software change, no GPU upgrade, no model compression
        removes this physical floor. <br/><br/>
        <strong>Smaller models</strong> and <strong>faster GPUs</strong> reduce
        <em>compute time</em>, but the round-trip latency is dominated by
        <em>propagation delay</em> — they don't help. <br/><br/>
        <strong>Moving the server physically closer</strong> or
        <strong>deploying directly on the vehicle</strong> are the only solutions
        because they reduce the distance the signal must travel.
        This is why Edge ML exists as a deployment paradigm — not as a preference,
        but as a physical necessity.
    </div>
    """

    _title = "✅ Exactly right." if _exactly_right else (
        "⚠️ Partially right — review the highlighted options." if not _has_wrong else
        "⚠️ Not quite — some selections add compute speed, not reduce propagation delay."
    )
    _border = "#16a34a" if _exactly_right else ("#f59e0b" if not _has_wrong else "#ef4444")
    _bg_outer = "#f0fdf4" if _exactly_right else ("#fffbeb" if not _has_wrong else "#fef2f2")

    # The physics-violation callout reinforces *why* cloud-side fixes don't
    # work. It reads as scolding when the student already picked the two
    # correct answers, so we suppress it on an exactly-right submission and
    # let the prose explanation + math peek carry the point (#1305).
    _items = [
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
            "**INFEASIBLE — Cloud inference violates physics.**\n\n"
            "Distance: 2,000 km | Speed in fiber: ~200,000 km/s | "
            "Round-trip: 2 × 2,000 / 200,000 = **20 ms** | "
            "AV SLA: 10 ms | **Verdict: physically impossible.** "
            "No GPU upgrade, no model compression, no software optimization "
            "can fix this. The model must move to the vehicle."
        ), kind="danger"))

    _items.append(mo.accordion({
        "Math Peek: Propagation Delay": mo.md("""
    **Formula:**
    $$
    t_{\\text{round-trip}} = \\frac{2d}{c \\cdot n}
    $$

    **Variables:**
    - **d**: distance between client and server (km)
    - **c**: speed of light in vacuum (299,792 km/s)
    - **n**: fiber refractive index factor (~0.67)
    - At d = 2,000 km: t = 2 × 2,000 / (299,792 × 0.67) ≈ 20 ms — exceeds 10 ms SLA by 2x
    """)
    }))

    mo.vstack(_items)
    return

# ─── CONCEPT 3: THE DEPLOYMENT REGIMES ────────────────────────────────────────

@app.cell
def _(CANONICAL_TRACKS, check1, check2empty, mo):
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
        _constraints = ", ".join(_profile.dominant_constraints)
        _metrics = ", ".join(_profile.primary_metrics)
        _guardrails = ", ".join(_profile.guardrail_metrics)
        _cards += f"""
            <div style="background: white; border: 1px solid {_color}44; border-radius: 12px; padding: 20px;">
                <div style="font-weight: 800; color: #1e293b; font-size: 1.0rem; margin-bottom: 4px;">
                    {_profile.label}
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
                    <strong>Primary metrics:</strong> {_metrics}<br/>
                    <strong>Guardrails:</strong> {_guardrails}<br/>
                    <strong>Dominant constraints:</strong> {_constraints}
                </div>
            </div>
        """

    mo.vstack([
        mo.md("---"),
        mo.md("""
        ## The Four Canonical Tracks

        The physical constraints above do not create a generic continuum. In this
        course they resolve into four canonical tracks. You will choose one as
        your recurring point of view. Later labs will change the story, metrics,
        and decisions automatically from that choice.
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

# ─── CHECK 3 (constraint reasoning) ───────────────────────────────────────────

@app.cell
def _(check1, check2empty, mo):
    mo.stop(check1.value is None or check2empty())

    check3 = mo.ui.radio(
        options={
            "A)  Cloud ML — access to the most compute": "A",
            "B)  Edge ML — low latency and local processing": "B",
            "C)  Mobile ML — runs on the patient's own device": "C",
            "D)  TinyML — lowest power, can run for months on a battery": "D",
        },
        label="""**Check your understanding.** A hospital wants to deploy an AI system
    that detects sepsis from ICU sensor readings. Requirements: results within 2 ms of
    each sensor reading, no patient data can leave the hospital network, and the sensor
    node must run for 6 months on a small battery without replacement.

    Which deployment paradigm is the *only* one that satisfies all three requirements simultaneously?""",
    )

    mo.vstack([
        check3,
    ])
    return (check3,)

@app.cell
def _(check1, check2empty, check3, mo):
    mo.stop(check1.value is None or check2empty() or check3.value is None)

    _correct = check3.value == "D"
    _feedback = {
        "A": (
            "**Not quite.** Cloud ML violates two of the three requirements. "
            "Round-trip latency to a cloud server is 10–500 ms — far above the 2 ms requirement. "
            "And patient data would leave the hospital network, violating the privacy constraint. "
            "Cloud gives you power, but power is not the binding constraint here."
        ),
        "B": (
            "**Closer, but not sufficient.** Edge ML achieves low latency and local processing, "
            "satisfying the first two requirements. But an edge server draws tens of watts "
            "continuously — it cannot run for 6 months on a small battery. "
            "The power constraint eliminates it. Edge is right for latency; wrong for energy."
        ),
        "C": (
            "**Not quite.** Mobile ML runs locally (satisfying privacy) and can meet the "
            "latency target, but sustained operation at smartphone-level power draws "
            "3–5 W. A small sensor battery would last hours, not months. "
            "The energy envelope makes mobile ML infeasible for always-on sensing."
        ),
        "D": (
            "**Correct.** TinyML is the only paradigm that satisfies all three simultaneously. "
            "Inference happens directly on the sensor node — no network latency, no data "
            "leaving the hospital. Microcontrollers running at microwatts can sustain "
            "always-on sensing for months on a coin-cell battery. "
            "The model must fit in kilobytes — that is the engineering challenge this regime imposes. "
            "Notice: this was not a software preference. It was a constraint analysis."
        ),
    }

    mo.vstack([
        mo.callout(
            mo.md(_feedback[check3.value]),
            kind="success" if _correct else "warn",
        ),
    ])
    return

# ═══════════════════════════════════════════════════════════════════════════════
# ZONE C: INTERFACE ORIENTATION
# ═══════════════════════════════════════════════════════════════════════════════

# _act_why: "Before Lab 01, students should recognize the recurring case -> guess -> evidence -> decision rhythm without advanced terminology."

# ─── INTERFACE ORIENTATION INTRO ───────────────────────────────────────────────

@app.cell
def _(check1, check2empty, check3, mo):
    mo.stop(check1.value is None or check2empty() or check3.value is None)

    mo.vstack([
        mo.md("---"),
        mo.md("""
        ## How Every Lab in This Curriculum Is Structured

        Starting from Lab 01, every lab follows the same simple rhythm:
        read a short case, make a prediction, change a few controls, inspect
        the evidence, and write the decision you would defend.

        Before you begin Lab 01, spend two minutes with the tour below. The goal
        is only to recognize the pattern. The technical vocabulary arrives later,
        one lab at a time.
        """),
    ])
    return

@app.cell
def _(
    COLORS, check2empty, mo, check1,
    check3,
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
                2 · Make A Guess
            </div>
            <div style="color:#166534; font-size:0.83rem; line-height:1.55;">
                Before looking at the result, you record what you think will
                happen. A guess is not graded as a trick question; it gives you
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

    # ── LIVE COMPONENT TOUR via mo.ui.tabs ────────────────────────────
    _tab_overview = mo.vstack([
        mo.md("""
        **Lab rhythm** — later labs repeat the same learning loop.

        The technical topic changes from lab to lab, but the student move stays
        familiar: read the case, make a guess, move the controls, inspect the
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
        **Controls change evidence.** A radio button that asks for your guess only
        records your expectation. Sliders and dropdowns are the controls that
        change plots, tables, and status cards.
        """),
        mo.hstack([
            mo.vstack([
                mo.md("**Example control**"),
                mo.ui.slider(start=0, stop=100, step=5, value=55, label="Evidence strength:"),
            ], gap=1),
            mo.Html(f"""
            <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px;
                        padding:16px; min-width:220px;">
                <div style="font-size:0.7rem; font-weight:700; color:#94a3b8;
                            text-transform:uppercase; margin-bottom:8px;">Evidence Preview</div>
                <div style="font-size:0.82rem; color:#475569; line-height:1.8;">
                    Current evidence: <strong style="color:{COLORS['BlueLine']}">medium</strong><br/>
                    Track constraint: <strong style="color:{COLORS['GreenLine']}">protected</strong><br/>
                    Decision status: <strong style="color:{COLORS['OrangeLine']}">needs rationale</strong>
                </div>
                <div style="margin-top:10px; font-size:0.72rem; color:#94a3b8; font-style:italic;">
                    In later labs, real plots and<br/>tables update as you move controls.
                </div>
            </div>
            """),
        ], gap=2, justify="start"),
        mo.callout(
            mo.md("Use guesses to learn from surprise. Use controls to create the evidence you will reason from."),
            kind="warn",
        ),
    ])

    _tab_prediction = mo.vstack([
        mo.md("""
        **Guess first, then test.**

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
                <strong>Guess:</strong> What do you think is the most likely reason?
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
        "Guess First":      _tab_prediction,
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
                You now know the repeated loop: case, guess, controls, evidence,
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
        mo.md("_Complete all three checks above to unlock your deployment context selection._")
    )

    mo.vstack([
        mo.md("---"),
        mo.md("""
        ## Scenario Brief

        You have now seen why deployment context is a first-order engineering decision,
        not an afterthought. For the next 15 labs, you will carry one deployment context
        as your primary lens — the physical regime whose constraints will test every
        optimization technique you learn.

        **This is not a career choice.** It is a choice of which physical law will
        be your primary adversary. You will understand all four regimes —
        but you will develop deep intuition for one.

        ## Your Track

        Choose one canonical track. Later labs will read this choice from the
        local Design Ledger and adapt narrative, hardware references, metrics,
        guardrails, and report framing automatically.
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

# ─── CONTEXT REVEAL + STAKEHOLDER MESSAGE + LEDGER INIT ───────────────────────

@app.cell(hide_code=True)
def _(DecisionLog):
    decision_input, decision_ui = DecisionLog()
    return (decision_ui,)

@app.cell
def _(
    COLORS,
    check1,
    check2empty,
    check2value_list,
    check3,
    context_selector,
    decision_ui,
    get_track_profile,
    lab_metadata,
    ledger,
    mo,
    track_context,
        track_arc_context,
):
    mo.stop(
        check1.value is None
        or check2empty()
        or check3.value is None
        or context_selector.value is None,
        mo.vstack([
            context_selector,
            mo.md("_Select your deployment context above._"),
        ])
    )

    _track_id = context_selector.value
    _track_profile = get_track_profile(_track_id)
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
        "check1_answer":      check1.value,
        "check1_correct":     check1.value == "C",
        "check2_selections":  check2value_list(),
        "check3_answer":      check3.value,
        "check3_correct":     check3.value == "D",
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
        context_selector,
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
                {_track_profile.label} · {_track_profile.stakeholder}
            </div>
            <div style="font-size:0.88rem; color:#475569; margin-bottom:4px; line-height:1.5;">
                <strong>North Star:</strong> {_track_profile.narrative}
            </div>
            <div style="font-size:0.88rem; margin-bottom:18px; line-height:1.5;">
                <strong style="color:{_t['color']};">Arch Nemesis:</strong>
                <span style="color:#334155;"> {", ".join(_track_profile.dominant_constraints)}</span>
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
                f"**Design Ledger initialized** — track: `{_track_id}`. "
                "Your track pre-loads hardware defaults and scenario constraints "
                "in every lab from Lab 01 onward. Proceed to **Lab 01: ML Introduction**."
            ),
            kind="success",
        ),
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
    get_track_profile,
    lab_big_takeaways,
    lab_learning_objectives,
    lab_metadata,
    mo,
    report_export_panel,
):
    mo.stop(
        check1.value is None
        or check2empty()
        or check3.value is None
        or context_selector.value is None,
        mo.md("_Complete the checks and choose your track to unlock the local report._"),
    )

    _track_id = context_selector.value
    _profile = get_track_profile(_track_id)
    _selected = f"{_profile.label} ({_profile.category})"
    _report = build_lab_report(
        lab_metadata,
        track=_profile.label,
        scenario="Lab 00 track selection and lab-interface orientation",
        learning_objectives=lab_learning_objectives,
        predictions={
            "production_ml_domain": check1.value,
            "physical_constraint_actions": ", ".join(check2value_list()),
            "always_on_sensing_regime": check3.value,
        },
        evidence_summary={
            "selected_track": _selected,
            "hardware_ref": _profile.hardware_ref,
            "system_ref": _profile.system_ref or "single-device profile",
            "dominant_constraints": ", ".join(_profile.dominant_constraints),
        },
        final_decision=f"Use {_profile.label} as the student's canonical track for subsequent labs.",
        big_takeaways=lab_big_takeaways,
        reflections={
            "diagnosis": "Deployment context determines which constraints matter first.",
            "tradeoff": f"{_profile.label} emphasizes {_profile.primary_metrics[0]} while preserving {_profile.guardrail_metrics[0]}.",
            "residual_risk": "This orientation report records track intent; later labs must validate feasibility with MLSysIM solver outputs.",
        },
        residual_risk=(
            "Lab 00 establishes the track and source references. Later labs may expose "
            "additional constraints that change a specific design choice inside the same track."
        ),
        source_trace={
            "track_id": _profile.track_id,
            "hardware_ref": _profile.hardware_ref,
            "system_ref": _profile.system_ref or "single-device profile",
            "source_policy": _profile.source_policy,
            "report_builder": "mlsysbook_labs.build_lab_report",
        },
        result_snapshot={
            "track_id": _profile.track_id,
            "track_label": _profile.label,
            "hardware_ref": _profile.hardware_ref,
            "system_ref": _profile.system_ref,
            "check1_correct": check1.value == "C",
            "check3_correct": check3.value == "D",
        },
    )

    mo.vstack([
        mo.md("## Download Report"),
        mo.callout(
            mo.md(
                "Your Lab 00 report is generated locally from the current notebook state. "
                "The fallback text panel contains the same Markdown artifact if browser download fails."
            ),
            kind="info",
        ),
        report_export_panel(_report),
    ])
    return

# ─── CELL 20: SYNTHESIS ────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(COLORS, mo):
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
                    <strong>1. The model is the 5% &mdash; the infrastructure is the 95%.</strong>
                    A model that reaches 99% accuracy in a notebook is not a product until it runs
                    reliably, monitors its own degradation, and serves requests within physical
                    constraints. ML Systems Engineering optimizes the 95% that makes deployment possible.
                </div>
                <div style="margin-bottom: 10px;">
                    <strong>2. Physical constraints partition deployment into four incommensurable regimes.</strong>
                    The speed of light, thermodynamics, and memory physics create four distinct operating
                    envelopes &mdash; Cloud, Edge, Mobile, TinyML &mdash; that no software engineering
                    can bridge. Choosing the wrong regime makes the system impossible, not just slow.
                </div>
                <div>
                    <strong>3. Every later lab repeats the same learning loop.</strong>
                    You will read a case, make a guess, explore evidence, decide, and
                    write a short report. That repeated structure lets the technical
                    ideas get harder without making the interface feel new each time.
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
    1. Which track (Cloud Fleet, iPhone, RoboTaxi, or Oura Ring) did you choose, and what constraint will that track keep bringing back?

    2. Why is a model that works in a notebook not automatically a deployed ML system?

    3. What is the recurring order of work in later labs: what do you read, what do you guess, what controls do you move, what evidence do you inspect, and what do you report?

    *If you cannot answer all three from memory, revisit the track cards and interface orientation.*
    """)
        }),
    ])
    return

# ─── CELL 21: LEDGER_HUD ───────────────────────────────────────────────────────
@app.cell
def _(COLORS, get_track_profile, ledger, mo):
    _track   = ledger.get_track() or "NONE"
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
    _hud_status = "Uninitialized" if _track == "NONE" else "Active — Chapter 0"
    try:
        _hud_track = get_track_profile(_track).label
    except KeyError:
        _hud_track = _track.upper()

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
    def check2empty():
        return not (model_size.value or quantization.value or move_server.value or faster_gpu.value or edge_deploy.value)

    def check2value_list():
        check2values = []
        # have to use items directly here for dependency evaluation
        if (model_size.value or quantization.value or move_server.value or faster_gpu.value or edge_deploy.value):
            for item in ['model_size', 'quantization', 'move_server' ,'faster_gpu', 'edge_deploy']:
                if globals()[item].value:
                    check2values.append(item)
        return check2values

    return check2empty, check2value_list

if __name__ == "__main__":
    app.run()

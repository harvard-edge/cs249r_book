import marimo

__generated_with = "0.19.6"
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
#   2. Interface Orientation — cockpit anatomy, live levers, prediction lock, MathPeek
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

    # WASM bootstrap: install mlsysim from hosted wheel when running in browser
    if sys.platform == "emscripten":
        import micropip
        await micropip.install("https://mlsysbook.ai/labs/wheels/mlsysim-0.1.0-py3-none-any.whl")
    elif "mlsysim" not in sys.modules:
        _root = Path(__file__).resolve().parents[2]
        if str(_root) not in sys.path:
            sys.path.insert(0, str(_root))

    from mlsysim.labs.state import DesignLedger
    from mlsysim.labs.style import COLORS, LAB_CSS
    from mlsysim.labs.components import DecisionLog

    ledger = DesignLedger()
    return mo, ledger, COLORS, LAB_CSS


# ─── CELL 1: HEADER ────────────────────────────────────────────────────────────
@app.cell
def _(mo, LAB_CSS):
    mo.vstack([
        LAB_CSS,
        mo.md("""
        <div style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
                    padding: 36px 44px; border-radius: 16px; color: white;
                    box-shadow: 0 8px 32px rgba(0,0,0,0.3);">
            <div style="font-size: 0.72rem; font-weight: 700; letter-spacing: 0.18em;
                        color: #475569; text-transform: uppercase; margin-bottom: 10px;">
                Machine Learning Systems · Volume I · Lab 00
            </div>
            <h1 style="margin: 0 0 10px 0; font-size: 2.4rem; font-weight: 900;
                       color: #f8fafc; line-height: 1.1; letter-spacing: -0.02em;">
                The Architect's Portal
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
                    20–25 min
                </span>
                <span style="background: rgba(245,158,11,0.15); color: #fcd34d;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(245,158,11,0.25);">
                    No prior reading required
                </span>
            </div>
        </div>
        """),
    ])
    return


# ─── CELL 2: BRIEFING ──────────────────────────────────────────────────────────
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
                <div style="margin-bottom: 3px;">1. <strong>Identify why infrastructure accounts for 95% of a production ML system</strong> and why the remaining 5% (the model) cannot be the primary engineering concern.</div>
                <div style="margin-bottom: 3px;">2. <strong>Predict which deployment paradigm satisfies a given set of physical constraints</strong> (latency floor, power budget, memory capacity) from the four-regime framework.</div>
                <div style="margin-bottom: 3px;">3. <strong>Recognize each recurring UI component</strong> of the lab interface &mdash; prediction lock, Latency Waterfall, MathPeek accordion, and HUD footer &mdash; before encountering them in live labs.</div>
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
                    @sec-introduction and @sec-ml-systems.
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
def _(mo, check1):
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
def _(mo, check1):
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
                    London to New York = 36 ms minimum round-trip, one-way.
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

@app.cell
def _(mo, check1):
    mo.stop(check1.value is None)

    check2 = mo.ui.multiselect(
        options={
            "Use a smaller model with fewer parameters":   "model_size",
            "Apply INT8 quantization to reduce precision": "quantization",
            "Move the datacenter server physically closer": "move_server",
            "Use a faster GPU with higher TFLOPS":         "faster_gpu",
            "Deploy the model directly on the vehicle":    "edge_deploy",
        },
        label="""**Check your understanding.** An autonomous vehicle perception system
is routed to a cloud datacenter 2,000 km away. Round-trip latency is 40 ms.
The safety requirement is a 10 ms end-to-end decision loop.

Select **all approaches** that could actually solve the latency problem:""",
    )
    return (check2,)


@app.cell
def _(mo, check1, check2):
    mo.stop(check1.value is None or len(check2.value) == 0)

    _correct_set = {"move_server", "edge_deploy"}
    _selected = set(check2.value)
    _exactly_right = _selected == _correct_set
    _has_wrong     = bool(_selected - _correct_set)
    _missing_right = bool(_correct_set - _selected)

    _option_labels = {
        "model_size":   "Use a smaller model",
        "quantization": "Apply INT8 quantization",
        "move_server":  "Move the server physically closer",
        "faster_gpu":   "Use a faster GPU",
        "edge_deploy":  "Deploy on the vehicle",
    }

    _rows = ""
    for _key, _label in _option_labels.items():
        _is_selected = _key in _selected
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

    mo.vstack([
        check2,
        mo.Html(f"""
        <div style="background:{_bg_outer}; border:1.5px solid {_border};
                    border-radius:10px; padding:18px 20px; margin-top:8px;">
            <div style="font-weight:700; font-size:0.95rem; color:{_border}; margin-bottom:10px;">{_title}</div>
            {_rows}
            {_explanation}
        </div>
        """),
    ])
    return


# ─── CONCEPT 3: THE DEPLOYMENT REGIMES ────────────────────────────────────────

@app.cell
def _(mo, check1, check2):
    mo.stop(check1.value is None or len(check2.value) == 0)

    mo.vstack([
        mo.md("---"),
        mo.md("""
        ## The Four Physical Regimes

        The physical constraints above don't create a continuum — they create
        **four distinct operating envelopes**, each demanding different infrastructure,
        different optimization strategies, and different definitions of "correct."
        """),
        mo.Html("""
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; margin: 16px 0;">

            <div style="background: white; border: 1px solid #c7d2fe; border-radius: 12px; padding: 20px;">
                <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 10px;">
                    <span style="font-size: 1.6rem;">☁️</span>
                    <div>
                        <div style="font-weight: 800; color: #1e293b;">Cloud ML</div>
                        <div style="font-size: 0.78rem; color: #6366f1; font-weight: 600;">
                            Binding constraint: Memory Bandwidth Wall
                        </div>
                    </div>
                </div>
                <div style="color: #475569; font-size: 0.87rem; line-height: 1.6; margin-bottom: 12px;">
                    Virtually unlimited compute and storage. The binding constraint
                    is not processing power — it is how fast data can move from
                    memory to compute cores. Most large models are <em>memory-bandwidth-bound</em>,
                    not compute-bound.
                </div>
                <div style="background: #eef2ff; border-radius: 8px; padding: 8px 12px;
                            font-size: 0.8rem; color: #3730a3; font-weight: 600;">
                    Latency: 100–500 ms · Power: kilowatts · Memory: terabytes
                </div>
            </div>

            <div style="background: white; border: 1px solid #fecaca; border-radius: 12px; padding: 20px;">
                <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 10px;">
                    <span style="font-size: 1.6rem;">🤖</span>
                    <div>
                        <div style="font-weight: 800; color: #1e293b;">Edge ML</div>
                        <div style="font-size: 0.78rem; color: #ef4444; font-weight: 600;">
                            Binding constraint: Latency Determinism Wall
                        </div>
                    </div>
                </div>
                <div style="color: #475569; font-size: 0.87rem; line-height: 1.6; margin-bottom: 12px;">
                    Computation happens near the data source — factory floors,
                    vehicles, hospitals. The binding constraint is not average latency
                    but <em>tail latency</em>: a single spike in a safety-critical system
                    is a failure, not a statistic.
                </div>
                <div style="background: #fef2f2; border-radius: 8px; padding: 8px 12px;
                            font-size: 0.8rem; color: #991b1b; font-weight: 600;">
                    Latency: 10–100 ms · Power: watts–tens of watts · Memory: gigabytes
                </div>
            </div>

            <div style="background: white; border: 1px solid #fed7aa; border-radius: 12px; padding: 20px;">
                <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 10px;">
                    <span style="font-size: 1.6rem;">📱</span>
                    <div>
                        <div style="font-weight: 800; color: #1e293b;">Mobile ML</div>
                        <div style="font-size: 0.78rem; color: #f59e0b; font-weight: 600;">
                            Binding constraint: Thermal Power Wall
                        </div>
                    </div>
                </div>
                <div style="color: #475569; font-size: 0.87rem; line-height: 1.6; margin-bottom: 12px;">
                    Intelligence runs directly on consumer devices. Compute capability
                    is substantial, but sustained operation is limited by heat
                    accumulation in a sealed, handheld enclosure. After thermal
                    throttling, <em>performance drops by 30–70%</em>.
                </div>
                <div style="background: #fffbeb; border-radius: 8px; padding: 8px 12px;
                            font-size: 0.8rem; color: #92400e; font-weight: 600;">
                    Latency: 5–50 ms · Power: 3–5 W sustained · Memory: 4–16 GB
                </div>
            </div>

            <div style="background: white; border: 1px solid #bbf7d0; border-radius: 12px; padding: 20px;">
                <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 10px;">
                    <span style="font-size: 1.6rem;">👂</span>
                    <div>
                        <div style="font-weight: 800; color: #1e293b;">TinyML</div>
                        <div style="font-size: 0.78rem; color: #10b981; font-weight: 600;">
                            Binding constraint: SRAM Capacity Wall
                        </div>
                    </div>
                </div>
                <div style="color: #475569; font-size: 0.87rem; line-height: 1.6; margin-bottom: 12px;">
                    Always-on intelligence in microcontrollers running on
                    coin-cell batteries. There is no operating system, no virtual
                    memory, no paging. If the model does not fit in 256 KB of SRAM,
                    it does not run. <em>Every byte is a resource allocation decision</em>.
                </div>
                <div style="background: #ecfdf5; border-radius: 8px; padding: 8px 12px;
                            font-size: 0.8rem; color: #064e3b; font-weight: 600;">
                    Latency: 1–10 ms · Power: microwatts–milliwatts · Memory: kilobytes
                </div>
            </div>

        </div>
        """),
        mo.callout(
            mo.md(
                "**Nine orders of magnitude** separate the largest cloud deployment "
                "(megawatts, terabytes) from the smallest TinyML device (microwatts, kilobytes). "
                "The engineering principles that govern one end of this spectrum "
                "do not transfer to the other. This is why ML Systems is a discipline, "
                "not a configuration setting."
            ),
            kind="info",
        ),
    ])
    return


# ─── CHECK 3 (constraint reasoning) ───────────────────────────────────────────

@app.cell
def _(mo, check1, check2):
    mo.stop(check1.value is None or len(check2.value) == 0)

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
    return (check3,)


@app.cell
def _(mo, check1, check2, check3):
    mo.stop(check1.value is None or len(check2.value) == 0 or check3.value is None)

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

# _act_why: "Before Lab 01, students must recognize every recurring UI component — prediction locks, instruments, MathPeek — so cognitive load goes to content, not navigation."

# ─── INTERFACE ORIENTATION INTRO ───────────────────────────────────────────────

@app.cell
def _(mo, check1, check2, check3):
    mo.stop(check1.value is None or len(check2.value) == 0 or check3.value is None)

    mo.vstack([
        mo.md("---"),
        mo.md("""
        ## How Every Lab in This Curriculum Is Structured

        Starting from Lab 01, every lab follows the same **four-zone cockpit layout**.
        This is not aesthetic — it is a deliberate information architecture that
        separates *what you control* from *what the system tells you*.

        Before you begin Lab 01, spend two minutes with the interactive tour below.
        You will recognize every element the moment you see it.
        """),
    ])
    return


@app.cell
def _(mo, check1, check2, check3, COLORS):
    mo.stop(check1.value is None or len(check2.value) == 0 or check3.value is None)

    # ── ZONE ANATOMY DIAGRAM ─────────────────────────────────────────
    _zone_html = """
    <div style="display:grid; grid-template-columns:1fr 1fr; gap:14px; margin:16px 0;">

        <div style="background:#f0f4ff; border:1.5px solid #c7d2fe; border-radius:10px;
                    padding:16px; border-top:4px solid #6366f1;">
            <div style="font-weight:800; color:#3730a3; font-size:0.9rem; margin-bottom:6px;">
                Zone 1 · Command Header
            </div>
            <div style="color:#4338ca; font-size:0.83rem; line-height:1.55;">
                Lab number, scenario title, your active persona, and live
                <strong>constraint badges</strong> (Latency, Power, Memory).
                Badges turn red the moment you violate a constraint —
                the system doesn't wait to tell you. Always visible.
            </div>
        </div>

        <div style="background:#f0fdf4; border:1.5px solid #bbf7d0; border-radius:10px;
                    padding:16px; border-top:4px solid #16a34a;">
            <div style="font-weight:800; color:#14532d; font-size:0.9rem; margin-bottom:6px;">
                Zone 2 · Engineering Levers
            </div>
            <div style="color:#166534; font-size:0.83rem; line-height:1.55;">
                Sliders, dropdowns, and toggles that modify your design —
                hardware target, batch size, precision, model variant.
                <strong>Every change recalculates everything instantly.</strong>
                No "Submit" button. The lab reacts in real-time.
            </div>
        </div>

        <div style="background:#fff7ed; border:1.5px solid #fed7aa; border-radius:10px;
                    padding:16px; border-top:4px solid #ea580c;">
            <div style="font-weight:800; color:#9a3412; font-size:0.9rem; margin-bottom:6px;">
                Zone 3 · Live Telemetry
            </div>
            <div style="color:#7c2d12; font-size:0.83rem; line-height:1.55;">
                Metric cards, Roofline chart (from Lab 11), Latency Waterfall
                (from Lab 02). All charts <strong>update as you move sliders</strong>.
                Your job is to read these instruments and trace cause to effect.
            </div>
        </div>

        <div style="background:#fffbeb; border:1.5px solid #fde68a; border-radius:10px;
                    padding:16px; border-top:4px solid #d97706;">
            <div style="font-weight:800; color:#92400e; font-size:0.9rem; margin-bottom:6px;">
                Zone 4 · Audit Trail
            </div>
            <div style="color:#78350f; font-size:0.83rem; line-height:1.55;">
                Consequence log, explanatory text, and a free-form rationale box.
                <strong>Explain your design decision</strong> in writing before
                submitting. The act of articulating trade-offs is the learning —
                not the number the simulator returns.
            </div>
        </div>

    </div>
    """

    # ── LIVE COMPONENT TOUR via mo.ui.tabs ────────────────────────────
    _tab_overview = mo.vstack([
        mo.md("""
        **`mo.ui.tabs`** — labs with multiple acts use tab navigation.
        Each tab is a self-contained section. You are looking at a live example right now.

        In later labs, tabs structure the workflow:
        ```
        Act I: Baseline     →  establish the initial state
        Act II: Intervention →  apply an optimization
        ```
        The tab structure ensures you *commit* to a baseline before modifying it.
        This is not UX convenience — it enforces the scientific method: measure before you change.
        """),
        mo.callout(
            mo.md("Switch between tabs above to navigate. Your work in each tab is preserved independently."),
            kind="info"
        ),
    ])

    _tab_levers = mo.vstack([
        mo.md("**Zone 2 levers** update the system state reactively. Here is a live example:"),
        mo.hstack([
            mo.vstack([
                mo.md("**Hardware target**"),
                mo.ui.dropdown(
                    options=["H100 (Cloud)", "Jetson Orin NX (Edge)", "Smartphone NPU (Mobile)", "Cortex-M7 (TinyML)"],
                    value="H100 (Cloud)",
                    label="Select hardware:"
                ),
                mo.md("**Batch size**"),
                mo.ui.slider(start=1, stop=128, step=1, value=32, label="Batch size:"),
            ], gap=1),
            mo.Html(f"""
            <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px;
                        padding:16px; min-width:220px;">
                <div style="font-size:0.7rem; font-weight:700; color:#94a3b8;
                            text-transform:uppercase; margin-bottom:8px;">Live Telemetry Preview</div>
                <div style="font-size:0.82rem; color:#475569; line-height:1.8;">
                    Latency: <strong style="color:{COLORS['BlueLine']}">12.4 ms</strong><br/>
                    Throughput: <strong style="color:{COLORS['GreenLine']}">2,580 tok/s</strong><br/>
                    Memory: <strong style="color:{COLORS['OrangeLine']}">34.2 GB</strong><br/>
                    MFU: <strong style="color:{COLORS['RedLine']}">47%</strong>
                </div>
                <div style="margin-top:10px; font-size:0.72rem; color:#94a3b8; font-style:italic;">
                    In real labs these numbers<br/>update as you move sliders.
                </div>
            </div>
            """),
        ], gap=2, justify="start"),
        mo.callout(
            mo.md("**Key insight:** Every lever connects to every metric. Changing batch size affects memory, which affects throughput, which affects cost. The cockpit shows all effects simultaneously."),
            kind="warn",
        ),
    ])

    _tab_prediction = mo.vstack([
        mo.md("""
        **The Prediction Lock** — the most important component in the curriculum.

        Before every Act in Labs 01–14, you will see a **Prediction Lock** like the one below.
        You must commit to a prediction *before* you can run the simulation.
        """),
        mo.Html("""
        <div style="background:#1e293b; border-radius:10px; padding:20px; border-left:4px solid #6366f1;">
            <div style="font-size:0.72rem; font-weight:700; color:#6366f1;
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:10px;">
                🔒 Prediction Lock — Act I
            </div>
            <div style="color:#e2e8f0; font-size:0.9rem; line-height:1.6; margin-bottom:14px;">
                <strong>Scenario:</strong> You double the batch size from 32 to 64 on an H100.
                The model is memory-bandwidth-bound.<br/><br/>
                <strong>Predict:</strong> Will throughput (tokens/second) increase,
                decrease, or stay approximately the same?
            </div>
            <div style="display:flex; gap:12px; flex-wrap:wrap;">
                <div style="background:rgba(99,102,241,0.15); border:1px solid #6366f1;
                            border-radius:8px; padding:8px 16px; color:#a5b4fc; font-size:0.85rem;
                            font-weight:600; cursor:pointer;">
                    A) Increase proportionally (~2×)
                </div>
                <div style="background:rgba(99,102,241,0.15); border:1px solid #6366f1;
                            border-radius:8px; padding:8px 16px; color:#a5b4fc; font-size:0.85rem;
                            font-weight:600; cursor:pointer;">
                    B) Increase sub-linearly
                </div>
                <div style="background:rgba(99,102,241,0.15); border:1px solid #6366f1;
                            border-radius:8px; padding:8px 16px; color:#a5b4fc; font-size:0.85rem;
                            font-weight:600; cursor:pointer;">
                    C) Stay the same
                </div>
            </div>
            <div style="margin-top:12px; font-size:0.78rem; color:#64748b; font-style:italic;">
                ↑ In a real lab, selecting an answer here unlocks the simulation instruments below.
            </div>
        </div>
        """),
        mo.md("""
        **Why this matters:** Research on deliberate practice shows that making an
        explicit prediction before observing a result dramatically increases retention.
        If your prediction is wrong, you experience *productive failure* — the gap
        between expectation and observation drives deeper encoding than passive reading.

        The prediction lock is not a gatekeeping mechanism. It is a learning amplifier.
        """),
    ])

    _tab_mathpeek = mo.vstack([
        mo.md("""
        **`MathPeek` accordion** — the invariant behind every instrument.

        Every chart and metric in the telemetry panel connects to a physical equation.
        The MathPeek accordion surfaces that equation on demand — you are never just
        moving sliders, you are probing the underlying physics.
        """),
        mo.accordion({
            "📐 View the Invariant — Iron Law of ML Systems (Preview)": mo.md("""
            **Formula:** `T = D/BW + O/R + L`

            **Components:**
            - **T** — Total end-to-end latency (seconds)
            - **D** — Data size (bytes moved across memory hierarchy)
            - **BW** — Memory bandwidth (bytes/second)
            - **O** — FLOPs required (floating-point operations)
            - **R** — Compute rate (FLOPs/second, hardware peak × MFU)
            - **L** — Fixed overhead latency (dispatch tax, network RTT)

            _This equation is the central object of the entire curriculum.
            You will encounter it in every lab. Open this accordion whenever
            you need to re-anchor a number to first principles._
            """),
        }),
        mo.callout(
            mo.md("**Lab 01** introduces the Iron Law formally. For now, recognize the accordion — it lives in every lab."),
            kind="info",
        ),
    ])

    _tour_tabs = mo.ui.tabs({
        "🏗️ Cockpit Anatomy":  _tab_overview,
        "🎛️ Live Levers":      _tab_levers,
        "🔒 Prediction Lock":  _tab_prediction,
        "📐 MathPeek":         _tab_mathpeek,
    })

    mo.vstack([
        _tour_tabs,
        mo.Html("""
        <div style="background:#0f172a; border-radius:10px; padding:16px 22px; margin-top:16px;
                    border:1px solid #1e293b; display:flex; align-items:center; gap:16px;">
            <div style="font-size:1.3rem;">✅</div>
            <div style="font-size:0.87rem; color:#94a3b8; line-height:1.6;">
                <strong style="color:#e2e8f0;">Interface orientation complete.</strong>
                You now recognize the four-zone cockpit, the live lever pattern, the
                prediction lock, and the MathPeek accordion. These are the only UI
                primitives used across all 14 labs — nothing new will be introduced
                without explanation.
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
def _(mo, check1, check2, check3):
    mo.stop(
        check1.value is None or len(check2.value) == 0 or check3.value is None,
        mo.md("_Complete all three checks above to unlock your deployment context selection._")
    )

    mo.vstack([
        mo.md("---"),
        mo.md("""
        ## Choose Your Physical Regime

        You have now seen why deployment context is a first-order engineering decision,
        not an afterthought. For the next 15 labs, you will carry one deployment context
        as your primary lens — the physical regime whose constraints will test every
        optimization technique you learn.

        **This is not a career choice.** It is a choice of which physical law will
        be your primary adversary. You will understand all four regimes —
        but you will develop deep intuition for one.
        """),
    ])
    return


@app.cell
def _(mo, check1, check2, check3):
    mo.stop(check1.value is None or len(check2.value) == 0 or check3.value is None)

    context_selector = mo.ui.radio(
        options={
            "☁️  Cloud ML  — your constraint is the Memory Bandwidth Wall":          "cloud",
            "🤖  Edge ML   — your constraint is the Latency Determinism Wall":        "edge",
            "📱  Mobile ML — your constraint is the Thermal Power Wall":              "mobile",
            "👂  TinyML    — your constraint is the SRAM Capacity Wall":              "tiny",
        },
        label="Select the deployment regime you will focus on throughout this curriculum:",
    )
    return (context_selector,)


# ─── CONTEXT REVEAL + STAKEHOLDER MESSAGE + LEDGER INIT ───────────────────────

@app.cell
@app.cell(hide_code=True)
def _(mo):
    decision_input, decision_ui = DecisionLog()
    return decision_input, decision_ui


@app.cell
def _(mo, check1, check2, check3, context_selector, ledger, COLORS, decision_input, decision_ui):
    mo.stop(
        check1.value is None
        or len(check2.value) == 0
        or check3.value is None
        or context_selector.value is None,
        mo.vstack([
            context_selector,
            mo.md("_Select your deployment context above._"),
        ])
    )

    _key = context_selector.value
    _contexts = {
        "cloud": {
            "color":     COLORS["BlueLine"],
            "bg":        COLORS["BlueL"],
            "label":     "Cloud ML",
            "nemesis":   "Memory Bandwidth Wall",
            "role":      "LLM Infrastructure Lead",
            "north_star":"Maximize sustained serving throughput for a 70B-parameter model on a multi-GPU cluster.",
            "persona":   "Your CTO",
            "quote": (
                "We're burning $40,000 a day on GPU rentals. "
                "If hardware utilization doesn't hit 50% by next quarter, "
                "we run out of runway. The model is fine. The infrastructure is not. Fix it."
            ),
            "arc": [
                ("Labs 01–04", "Foundations",
                 "Learn the D·A·M taxonomy, the Iron Law, and why the Memory Wall is your primary constraint"),
                ("Labs 05–08", "Build",
                 "Trace memory allocation through a transformer forward pass; profile your serving stack"),
                ("Labs 09–11", "Optimize",
                 "Apply quantization, understand hardware utilization, and cross the efficiency threshold"),
                ("Labs 12–14", "Deploy",
                 "Benchmark, monitor, and operate a production serving system at scale"),
            ],
        },
        "edge": {
            "color":     COLORS["RedLine"],
            "bg":        COLORS["RedL"],
            "label":     "Edge ML",
            "nemesis":   "Latency Determinism Wall",
            "role":      "Autonomous Systems Lead",
            "north_star":"Maintain a deterministic 10 ms perception-to-decision loop on a Jetson Orin NX.",
            "persona":   "Your Safety Director",
            "quote": (
                "A 5 ms latency spike added 15 cm of stopping distance at 60 mph. "
                "That is a regulatory failure. I do not care about your average latency. "
                "One tail event is one too many. Zero tolerance."
            ),
            "arc": [
                ("Labs 01–04", "Foundations",
                 "Understand latency decomposition, the Iron Law, and why P99 is the only metric that matters"),
                ("Labs 05–08", "Build",
                 "Implement a priority scheduler; measure the tail-latency distribution of your inference stack"),
                ("Labs 09–11", "Optimize",
                 "Apply structured pruning to reduce worst-case latency below the safety threshold"),
                ("Labs 12–14", "Deploy",
                 "Validate deterministic SLAs on physical edge hardware under adversarial load"),
            ],
        },
        "mobile": {
            "color":     COLORS["OrangeLine"],
            "bg":        COLORS["OrangeL"],
            "label":     "Mobile ML",
            "nemesis":   "Thermal Power Wall",
            "role":      "Smartphone App Architect",
            "north_star":"Run 60 FPS real-time on-device inference within a 2 W sustained thermal envelope.",
            "persona":   "Your UX Director",
            "quote": (
                "Users are returning the device because it heats up after two minutes of AR. "
                "You have 2 Watts of sustained thermal headroom. Not 2.1. Two. "
                "Every watt you save is a feature."
            ),
            "arc": [
                ("Labs 01–04", "Foundations",
                 "Map the D·A·M trade-off for mobile NPUs; quantify the thermal budget"),
                ("Labs 05–08", "Build",
                 "Implement MobileNetV2 with depthwise separable convolutions in TinyTorch"),
                ("Labs 09–11", "Optimize",
                 "Apply INT8 quantization and operator fusion to stay within the thermal envelope"),
                ("Labs 12–14", "Deploy",
                 "Benchmark sustained throughput on a power-constrained device under realistic workloads"),
            ],
        },
        "tiny": {
            "color":     COLORS["GreenLine"],
            "bg":        COLORS["GreenL"],
            "label":     "TinyML",
            "nemesis":   "SRAM Capacity Wall",
            "role":      "TinyML / Embedded Systems Lead",
            "north_star":"Fit real-time keyword spotting in under 256 KB SRAM, running under 1 mW.",
            "persona":   "Your Hardware Lead",
            "quote": (
                "We have 256 KB of on-chip SRAM. Every weight byte you keep "
                "is audio buffer you lose. There is no paging. There is no swap. "
                "If it does not fit, it does not run."
            ),
            "arc": [
                ("Labs 01–04", "Foundations",
                 "Count every byte in a DS-CNN keyword spotting model; understand SRAM allocation"),
                ("Labs 05–08", "Build",
                 "Implement depthwise separable convolutions in TinyTorch; profile memory layout"),
                ("Labs 09–11", "Optimize",
                 "Achieve 4× compression via magnitude pruning and INT8 quantization"),
                ("Labs 12–14", "Deploy",
                 "Fit the full inference pipeline in 256 KB and validate on a physical MCU"),
            ],
        },
    }

    _t = _contexts[_key]

    # Persist to Design Ledger
    ledger.save(chapter=0, design={
        "deployment_context": _key,
        "check1_answer":      check1.value,
        "check1_correct":     check1.value == "C",
        "check2_selections":  list(check2.value),
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
                {_t['label']} · {_t['role']}
            </div>
            <div style="font-size:0.88rem; color:#475569; margin-bottom:4px; line-height:1.5;">
                <strong>North Star:</strong> {_t['north_star']}
            </div>
            <div style="font-size:0.88rem; margin-bottom:18px; line-height:1.5;">
                <strong style="color:{_t['color']};">Arch Nemesis:</strong>
                <span style="color:#334155;"> {_t['nemesis']}</span>
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
                f"**Design Ledger initialized** — context: `{_key}`. "
                "Your deployment regime pre-loads hardware defaults and scenario constraints "
                "in every lab from Lab 01 onward. Proceed to **Lab 01: ML Introduction**."
            ),
            kind="success",
        ),
    ])
    return


# ─── CELL 20: SYNTHESIS ────────────────────────────────────────────────────────
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
                    <strong>3. The prediction lock, Latency Waterfall, and MathPeek accordion appear in every lab.</strong>
                    These are not UX choices &mdash; the prediction lock enforces scientific method,
                    the Waterfall shows Iron Law terms in real time, and the MathPeek surfaces the
                    governing equation behind every number. Recognize them before Lab 01.
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
                    <strong>Lab 01: The Magnitude Awakening.</strong> This lab established
                    that four deployment regimes exist and that constraints are incommensurable.
                    Lab 01 asks: by exactly how many orders of magnitude do they differ &mdash;
                    and does that gap force separate software stacks or merely require tuning?
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
                    <strong>Read:</strong> @sec-introduction for the 95% Problem, the D&middot;A&middot;M
                    Taxonomy, and the four deployment paradigms with their physical constraints.<br/>
                    <strong>Build:</strong> TinyTorch starts in Module 01 &mdash; the foundations
                    module that builds the forward-pass engine you will profile throughout the curriculum.
                </div>
            </div>

        </div>
        """),

        mo.accordion({
            "Self-Assessment: Can you answer these?": mo.md("""
    1. Which deployment context (cloud, mobile, edge, or TinyML) did you choose, and what was the binding constraint that defined your track — compute, memory, power, or latency?

    2. The D·A·M Triad claims that Data, Algorithm, and Machine are inseparable. After selecting your track, which axis most constrained what model architectures were feasible?

    3. If a friend claims 'the same trained model can run on any device — you just need to make it smaller,' what specific numbers from Act I would you cite to explain why a 2,000,000x compute gap and a 160,000x memory gap make this impossible in practice?

    *If you cannot answer all three from memory, revisit Act I and Act II.*
    """)
        }),
    ])
    return


# ─── CELL 21: LEDGER_HUD ───────────────────────────────────────────────────────
@app.cell
def _(mo, ledger, COLORS):
    _track   = ledger.get_track() or "NONE"
    _color_map = {
        "cloud":  COLORS["BlueLine"],
        "edge":   COLORS["RedLine"],
        "mobile": COLORS["OrangeLine"],
        "tiny":   COLORS["GreenLine"],
        "NONE":   "#475569",
    }
    _hud_color  = _color_map.get(_track, "#475569")
    _hud_status = "Uninitialized" if _track == "NONE" else "Active — Chapter 0"

    mo.Html(f"""
    <div style="display:flex; gap:28px; align-items:center; padding:12px 24px;
                background:#0f172a; border-radius:10px; margin-top:32px;
                font-family:'SF Mono','Fira Code',monospace; font-size:0.8rem;
                border:1px solid #1e293b;">
        <div style="color:#475569; font-weight:600; letter-spacing:0.06em;">🗂️ DESIGN LEDGER</div>
        <div>
            <span style="color:#475569;">Context: </span>
            <span style="color:{_hud_color}; font-weight:700;">{_track.upper()}</span>
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


if __name__ == "__main__":
    app.run()

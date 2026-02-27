import marimo

__generated_with = "0.10.15"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    import sys
    import os
    from pathlib import Path
    import plotly.graph_objects as go
    import numpy as np

    try:
        notebook_path = Path(os.path.abspath(__file__))
    except NameError:
        notebook_path = Path(os.getcwd()) / "labs" / "vol1" / "lab_00_the_map.py"

    project_root = notebook_path.parents[2]
    sys.path.append(str(project_root / "book" / "quarto"))
    sys.path.append(str(project_root))

    from labs.core.style import COLORS, LAB_CSS, apply_plotly_theme
    from labs.core.components import Card, PredictionLock, MetricRow, StakeholderMessage, MathPeek
    from labs.core.state import DesignLedger

    ledger = DesignLedger()

    return (
        COLORS,
        Card,
        DesignLedger,
        LAB_CSS,
        MathPeek,
        MetricRow,
        PredictionLock,
        StakeholderMessage,
        apply_plotly_theme,
        go,
        ledger,
        mo,
        np,
        os,
        project_root,
        sys,
    )


@app.cell
def __(LAB_CSS, mo):
    mo.vstack([
        LAB_CSS,
        mo.md("# 🗺️ Lab 00: The Architect's Portal"),
        mo.md(r"""
        ### The Question Every Engineer Ignores

        Before you write a single line of code, you are making architectural decisions.
        Engineers who discover **memory constraints** during the *Deploy* phase instead of the *Design* phase
        pay a correction cost **10–100× higher** than those who caught it in simulation.

        This lab is your orientation. Three missions. Three instruments. One irreversible track choice
        that will define the next 16 chapters.

        **Complete the missions in order.**
        """),
        mo.md("---"),
    ])


@app.cell
def __(mo):
    kat_tabs = mo.ui.tabs({
        "1. ECOSYSTEM AUDIT": mo.md(""),
        "2. GEARBOX CERT": mo.md(""),
        "3. CLAIM MISSION": mo.md(""),
    })
    kat_tabs
    return (kat_tabs,)


@app.cell
def __(PredictionLock, ledger, mo):
    # ── KAT 1: ECOSYSTEM AUDIT ────────────────────────────────────────────────
    kat1_predict, kat1_lock_ui = PredictionLock(
        1,
        "If a memory constraint is discovered during Deploy (not Design), "
        "by what factor does the cost of fixing the mistake increase? "
        "(Give a ratio, e.g. 10x)",
    )
    kat1_read_slider = mo.ui.slider(
        start=0, stop=60, value=25, step=5, label="📖 Read (Textbook) %"
    )
    kat1_design_slider = mo.ui.slider(
        start=0, stop=60, value=25, step=5, label="🔬 Design (Labs) %"
    )
    kat1_build_slider = mo.ui.slider(
        start=0, stop=60, value=25, step=5, label="🔨 Build (TinyTorch) %"
    )
    kat1_reflect = mo.ui.text_area(
        label="REFLECT: Justify why 10 hours in the Lab (Design) can save 100 hours "
              "in TinyTorch (Build). Reference the 'cost of late discovery.'",
        full_width=True,
    )

    # ── KAT 2: GEARBOX CERTIFICATION ─────────────────────────────────────────
    kat2_predict, kat2_lock_ui = PredictionLock(
        2,
        "If you double the clock frequency (f) of a processor, does the power draw (P) "
        "double, quadruple, or increase by 8x (cubically)? Type your prediction.",
    )
    kat2_freq_slider = mo.ui.slider(
        start=0.5, stop=5.0, value=1.0, step=0.1, label="Clock Frequency (GHz)"
    )
    kat2_reflect = mo.ui.text_area(
        label="REFLECT: Why is 'Slider Guessing' dangerous in systems engineering? "
              "How did the Prediction step change your relationship with the chart?",
        full_width=True,
    )

    # ── KAT 3: CLAIM MISSION ──────────────────────────────────────────────────
    _initial_track = ledger.get_track()
    if _initial_track == "NONE":
        _initial_track = None

    kat3_track_radio = mo.ui.radio(
        options={
            "☁️  Cloud Titan — LLM Serving Architect": "CLOUD",
            "🤖  Edge Guardian — AV Systems Lead": "EDGE",
            "🕶️  Mobile Nomad — AR Glasses Developer": "MOBILE",
            "👂  Tiny Pioneer — Neural Hearable Lead": "TINY",
        },
        value=_initial_track,
        label="Select your Career Specialization:",
    )
    kat3_reflect = mo.ui.text_area(
        label="REFLECT: In your own words, describe the physical barrier that stands "
              "between you and your North Star mission.",
        full_width=True,
    )

    return (
        kat1_predict,
        kat1_lock_ui,
        kat1_read_slider,
        kat1_design_slider,
        kat1_build_slider,
        kat1_reflect,
        kat2_predict,
        kat2_lock_ui,
        kat2_freq_slider,
        kat2_reflect,
        kat3_track_radio,
        kat3_reflect,
    )


@app.cell
def __(
    Card,
    COLORS,
    MathPeek,
    MetricRow,
    kat1_build_slider,
    kat1_design_slider,
    kat1_lock_ui,
    kat1_predict,
    kat1_read_slider,
    kat1_reflect,
    kat_tabs,
    mo,
):
    # ── KAT 1 RENDER ─────────────────────────────────────────────────────────
    def render_kat1():
        mo.stop(kat_tabs.value != "1. ECOSYSTEM AUDIT")

        _header = mo.md(r"""
        ## KAT 1: The Ecosystem Audit

        The ML Systems curriculum has four phases, each serving a distinct purpose:

        | Phase | Tool | Purpose |
        |-------|------|---------|
        | **Read** | Textbook | Theory — understand *why* systems work |
        | **Design** | These Labs | Analysis — explore *trade-offs* in simulation |
        | **Build** | TinyTorch | Mechanics — implement *how* frameworks work |
        | **Deploy** | Hardware Kits | Reality — confront *physical constraints* |

        **The Architect's Rule:** A bug caught in **Design** costs **1×** to fix.
        The same bug caught in **Deploy** costs **10–100×**.
        This is not opinion — it is empirical software engineering data.

        **Mission:** Allocate your 100-point engineering budget across Read, Design, and Build.
        Watch what happens to the Deploy phase when Design is underfunded.
        """)

        if kat1_predict.value == "":
            return mo.vstack([_header, kat1_lock_ui])

        _read_pct = kat1_read_slider.value
        _design_pct = kat1_design_slider.value
        _build_pct = kat1_build_slider.value
        _deploy_pct = max(0, 100 - _read_pct - _design_pct - _build_pct)
        _total_used = _read_pct + _design_pct + _build_pct

        _design_ok = _design_pct >= 15
        _over_budget = _total_used > 100

        _deploy_color = COLORS['GreenLine'] if (_design_ok and not _over_budget) else COLORS['RedLine']
        _deploy_bg = COLORS['GreenL'] if (_design_ok and not _over_budget) else COLORS['RedL']
        _deploy_label = "✅ Viable" if (_design_ok and not _over_budget) else "⚠️ VIOLATION"

        _pipeline_html = f"""
        <div style="display:flex; gap:8px; align-items:stretch; margin:1rem 0; font-size:0.85rem;">
          <div style="flex:1; padding:12px; background:{COLORS['BlueL']}; border:2px solid {COLORS['BlueLine']}; border-radius:8px; text-align:center;">
            <div style="font-weight:700; color:{COLORS['BlueLine']};">📖 READ</div>
            <div style="font-size:1.5rem; font-weight:800; color:{COLORS['BlueLine']};">{_read_pct}%</div>
            <div style="color:#777; font-size:0.75rem;">Textbook</div>
          </div>
          <div style="align-self:center; font-size:1.4rem; color:#aaa; padding:0 4px;">→</div>
          <div style="flex:1; padding:12px; background:{'#E8F5F0' if _design_ok else '#FFF3CD'}; border:2px solid {COLORS['GreenLine'] if _design_ok else COLORS['OrangeLine']}; border-radius:8px; text-align:center;">
            <div style="font-weight:700; color:{COLORS['GreenLine'] if _design_ok else COLORS['OrangeLine']};">🔬 DESIGN</div>
            <div style="font-size:1.5rem; font-weight:800; color:{COLORS['GreenLine'] if _design_ok else COLORS['OrangeLine']};">{_design_pct}%</div>
            <div style="color:{COLORS['GreenLine'] if _design_ok else COLORS['OrangeLine']}; font-size:0.75rem; font-weight:600;">{"✅ Funded" if _design_ok else "⚠️ UNDERFUNDED"}</div>
          </div>
          <div style="align-self:center; font-size:1.4rem; color:#aaa; padding:0 4px;">→</div>
          <div style="flex:1; padding:12px; background:{COLORS['OrangeL']}; border:2px solid {COLORS['OrangeLine']}; border-radius:8px; text-align:center;">
            <div style="font-weight:700; color:{COLORS['OrangeLine']};">🔨 BUILD</div>
            <div style="font-size:1.5rem; font-weight:800; color:{COLORS['OrangeLine']};">{_build_pct}%</div>
            <div style="color:#777; font-size:0.75rem;">TinyTorch</div>
          </div>
          <div style="align-self:center; font-size:1.4rem; color:#aaa; padding:0 4px;">→</div>
          <div style="flex:1; padding:12px; background:{_deploy_bg}; border:2px solid {_deploy_color}; border-radius:8px; text-align:center;">
            <div style="font-weight:700; color:{_deploy_color};">🚀 DEPLOY</div>
            <div style="font-size:1.5rem; font-weight:800; color:{_deploy_color};">{_deploy_pct}%</div>
            <div style="color:{_deploy_color}; font-size:0.75rem; font-weight:600;">{_deploy_label}</div>
          </div>
        </div>
        """

        if _over_budget:
            _status_msg = mo.Html(f"""
            <div style="background:{COLORS['RedL']}; border-left:4px solid {COLORS['RedLine']};
                        padding:1rem; border-radius:8px; margin:0.5rem 0;">
                <strong>⚠️ Over Budget:</strong> Read + Design + Build = {_total_used}%.
                Exceeds 100. Reduce one phase.
            </div>""")
        elif not _design_ok:
            _multiplier = max(10, 100 // max(1, _design_pct))
            _status_msg = mo.Html(f"""
            <div style="background:{COLORS['RedL']}; border-left:4px solid {COLORS['RedLine']};
                        padding:1rem; border-radius:8px; margin:0.5rem 0;">
                <strong>CONSTRAINT VIOLATION:</strong> Design budget is only {_design_pct}%.
                Constraints caught in Deploy cost approximately <strong>{_multiplier}×</strong>
                more than constraints caught in Design. The Deploy box is now in a failure state.
            </div>""")
        else:
            _status_msg = mo.Html(f"""
            <div style="background:{COLORS['GreenL']}; border-left:4px solid {COLORS['GreenLine']};
                        padding:1rem; border-radius:8px; margin:0.5rem 0;">
                <strong>System Viable:</strong> {_design_pct}% Design investment is sufficient
                for early constraint detection.
            </div>""")

        _cost_metrics = mo.Html("".join([
            MetricRow("Design Investment", f"{_design_pct}%", "≥15% required for viability"),
            MetricRow("Deploy Health", _deploy_label),
            MetricRow("Fix Cost if Constraint Hits Deploy", f"~{max(10, 100 // max(1, _design_pct))}×",
                      "vs. catching it in Design"),
        ]))

        return mo.vstack([
            _header,
            mo.md("#### 🎛️ Instrument: The Pipeline Budget Allocator"),
            mo.md("_Allocate Read, Design, and Build — remaining budget flows automatically to Deploy._"),
            mo.hstack([kat1_read_slider, kat1_design_slider, kat1_build_slider], gap=2),
            mo.Html(_pipeline_html),
            _status_msg,
            mo.hstack([
                Card("Budget Summary", _cost_metrics),
                Card("The Discovery Curve", mo.md(r"""
                | Phase | Relative Fix Cost |
                |-------|----------------:|
                | **Design** (Labs) | **1×** |
                | Build (TinyTorch) | ~10× |
                | Deploy (Hardware) | ~100× |

                *Cost-of-change models: Boehm (1981), updated by Capers Jones (2010).*
                Each phase that passes multiplies the correction cost by roughly 10×.
                """))
            ]),
            MathPeek(
                r"\text{Fix Cost} \propto 10^{\text{phase\_index}}",
                {
                    "phase_index": "0 = Design, 1 = Build, 2 = Deploy",
                    "Implication": "Every phase of delay multiplies correction cost by ~10×",
                }
            ),
            mo.md("---"),
            kat1_reflect,
        ])

    render_kat1()


@app.cell
def __(
    Card,
    COLORS,
    MathPeek,
    apply_plotly_theme,
    go,
    kat2_freq_slider,
    kat2_lock_ui,
    kat2_predict,
    kat2_reflect,
    kat_tabs,
    mo,
    np,
):
    # ── KAT 2 RENDER ─────────────────────────────────────────────────────────
    def render_kat2():
        mo.stop(kat_tabs.value != "2. GEARBOX CERT")

        _header = mo.md(r"""
        ## KAT 2: Gearbox Certification — The Physics of the Prediction

        Every performance chart in this curriculum represents a **physical law**, not a suggestion.
        Before you interact with any instrument, you must commit to a prediction.

        This task teaches you *why the Prediction step exists*. We use a simple physical system:
        the relationship between processor clock frequency ($f$) and power consumption ($P$).

        **The Law:** Dynamic power scales as $P = C \cdot V^2 \cdot f$.
        Under Dennard Scaling, voltage scaled proportionally with frequency ($V \propto f$),
        making $P \propto f^3$ — a **cubic** relationship.

        Before you move the slider: commit to your prediction.
        """)

        if kat2_predict.value == "":
            return mo.vstack([_header, kat2_lock_ui])

        _f = kat2_freq_slider.value
        _f_base = 1.0  # 1 GHz reference
        _P_base = 1.0  # normalized to 1W at 1 GHz

        _freqs = np.linspace(0.5, 5.0, 300)
        _power_linear = _P_base * (_freqs / _f_base)
        _power_quad   = _P_base * (_freqs / _f_base) ** 2
        _power_cubic  = _P_base * (_freqs / _f_base) ** 3

        _current_power = _P_base * (_f / _f_base) ** 3
        _double_power  = _P_base * ((_f * 2) / _f_base) ** 3
        _double_ratio  = _double_power / _current_power  # always 8

        _kat2_fig = go.Figure()
        _kat2_fig.add_trace(go.Scatter(
            x=_freqs, y=_power_linear, mode='lines',
            name='P ∝ f  (doubled f → 2× power)',
            line=dict(color=COLORS['Grey'], dash='dash', width=1.5),
        ))
        _kat2_fig.add_trace(go.Scatter(
            x=_freqs, y=_power_quad, mode='lines',
            name='P ∝ f²  (doubled f → 4× power)',
            line=dict(color=COLORS['OrangeLine'], dash='dot', width=1.5),
        ))
        _kat2_fig.add_trace(go.Scatter(
            x=_freqs, y=_power_cubic, mode='lines',
            name='P ∝ f³  (doubled f → 8× power) ← Real Law',
            line=dict(color=COLORS['RedLine'], width=3),
        ))
        # Operating point marker
        _kat2_fig.add_trace(go.Scatter(
            x=[_f], y=[_current_power], mode='markers',
            name=f'Your point ({_f:.1f} GHz)',
            marker=dict(size=14, color=COLORS['BlueLine'], symbol='diamond',
                        line=dict(color='white', width=2)),
        ))
        # Vertical reference line at selected frequency
        _kat2_fig.add_vline(
            x=_f, line_dash="dot", line_color=COLORS['BlueLine'], line_width=1,
        )

        _kat2_fig.update_layout(
            xaxis_title="Clock Frequency (GHz)",
            yaxis_title="Normalized Power (× baseline @ 1 GHz)",
            height=320,
            legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
        )

        _result_html = f"""
        <div style="display:flex; gap:1.5rem; flex-wrap:wrap; background:{COLORS['Neutral']};
                    border-radius:10px; padding:1rem; margin:0.5rem 0;">
          <div>
            <div style="color:#999; font-size:0.78rem; text-transform:uppercase; letter-spacing:0.05em;">
              SELECTED FREQUENCY</div>
            <div style="font-size:1.6rem; font-weight:800; color:{COLORS['BlueLine']};">
              {_f:.1f} GHz</div>
          </div>
          <div>
            <div style="color:#999; font-size:0.78rem; text-transform:uppercase; letter-spacing:0.05em;">
              NORMALIZED POWER</div>
            <div style="font-size:1.6rem; font-weight:800; color:{COLORS['RedLine']};">
              {_current_power:.2f}×</div>
          </div>
          <div>
            <div style="color:#999; font-size:0.78rem; text-transform:uppercase; letter-spacing:0.05em;">
              IF FREQUENCY DOUBLES →</div>
            <div style="font-size:1.6rem; font-weight:800; color:{COLORS['RedLine']};">
              Power × {int(round(_double_ratio))}</div>
          </div>
          <div>
            <div style="color:#999; font-size:0.78rem; text-transform:uppercase; letter-spacing:0.05em;">
              SCALING LAW</div>
            <div style="font-size:1.6rem; font-weight:800; color:{COLORS['RedLine']};">
              Cubic (f³)</div>
          </div>
        </div>
        """

        _insight = mo.md(r"""
        **Why does this matter for AI Engineering?**

        Modern accelerators run at near-fixed frequency precisely because of this law.
        The H100 runs at ~1.98 GHz (not 10 GHz) because beyond a thermal threshold,
        frequency scaling becomes physically uneconomical — each GHz costs 3× as much
        power as the last.

        Instead of faster clocks, hardware architects scaled **parallelism** (thousands of cores).
        This is not a preference — it is a consequence of the cubic power law you just witnessed.
        Every slider in this curriculum obeys laws like this one.
        """)

        return mo.vstack([
            _header,
            mo.md("#### 🎛️ Instrument: The Power Curve"),
            kat2_freq_slider,
            mo.Html(_result_html),
            Card("The Cubic Power Wall", mo.as_html(apply_plotly_theme(_kat2_fig))),
            MathPeek(
                r"P = C \cdot V^2 \cdot f \xrightarrow{V \propto f} P \propto f^3",
                {
                    "C": "Switched capacitance (chip design constant)",
                    "V": "Supply voltage (∝ f under Dennard Scaling)",
                    "f": "Clock frequency (GHz)",
                    "Key consequence": "Doubling f → 2³ = 8× power draw",
                }
            ),
            mo.md("---"),
            _insight,
            mo.md("---"),
            kat2_reflect,
        ])

    render_kat2()


@app.cell
def __(
    COLORS,
    MathPeek,
    StakeholderMessage,
    kat3_reflect,
    kat3_track_radio,
    kat_tabs,
    ledger,
    mo,
):
    # ── KAT 3 RENDER ─────────────────────────────────────────────────────────
    def render_kat3():
        mo.stop(kat_tabs.value != "3. CLAIM MISSION")

        _header = mo.md(r"""
        ## KAT 3: Claim Mission — Choose Your Physical Regime

        You have calibrated two instruments:
        - **KAT 1:** The cost of discovering constraints late (the Discovery Curve)
        - **KAT 2:** The physics of non-linear scaling (the Prediction)

        Now you choose the **physical regime** you will master across the next 16 chapters.
        Your choice defines your **Fixed North Star Mission** — the singular goal that every
        lab will push you toward — and your **Arch Nemesis**, the physical constraint that will
        appear in every chapter, increasingly harder to solve as the system scales.
        """)

        _tracks = {
            "CLOUD": {
                "role": "LLM Serving Architect",
                "goal": "Maximize Llama-2-70B serving throughput on a single H100 node.",
                "nemesis": "Memory Bandwidth Wall (HBM3)",
                "color": COLORS['BlueLine'],
                "bg": COLORS['BlueL'],
                "persona": "The Startup CFO",
                "quote": (
                    "We're burning $10k/day on H100 rentals. If GPU utilization isn't "
                    "above 80%, we run out of money by Chapter 13. Fix it, or we're done."
                ),
                "milestones": [
                    "Foundations: Understand the H100 Roofline and why Llama-2 is memory-bound at batch=1",
                    "Build: Implement KV-cache management and attention kernels in TinyTorch",
                    "Optimize: Apply INT8 quantization to fit larger batches within 80 GiB HBM",
                    "Deploy: Achieve 100+ tokens/s sustained throughput on a real H100 node",
                ],
            },
            "EDGE": {
                "role": "AV Systems Lead",
                "goal": "Maintain a deterministic 10ms perception-to-brake loop on NVIDIA Orin NX.",
                "nemesis": "Determinism Wall (Tail-Latency Jitter)",
                "color": COLORS['RedLine'],
                "bg": COLORS['RedL'],
                "persona": "The Safety Director",
                "quote": (
                    "A 5ms jitter spike means 15cm of extra stopping distance at 60 mph. "
                    "I don't care about average latency. One violation is a regulatory failure. "
                    "Zero tolerance."
                ),
                "milestones": [
                    "Foundations: Quantify the jitter budget using the Iron Law decomposition",
                    "Build: Implement a FIFO-priority inference scheduler in TinyTorch",
                    "Optimize: Apply structured pruning to achieve P99 latency < 8ms",
                    "Deploy: Validate deterministic SLAs on the Orin NX hardware kit",
                ],
            },
            "MOBILE": {
                "role": "AR Glasses Developer",
                "goal": "Run 60 FPS real-time translation overlay on Meta Ray-Bans under a 2W thermal cap.",
                "nemesis": "Thermal Wall (Power Density)",
                "color": COLORS['OrangeLine'],
                "bg": COLORS['OrangeL'],
                "persona": "The UX Director",
                "quote": (
                    "Users are returning the glasses because the frame burns their face after "
                    "2 minutes of AR. You have 2 Watts of thermal headroom. Not 2.1. Not 2.05. Two."
                ),
                "milestones": [
                    "Foundations: Map the thermal budget across the D·A·M axes for a mobile NPU",
                    "Build: Implement MobileNetV2 with depthwise separable convolutions in TinyTorch",
                    "Optimize: Apply INT8 quantization + operator fusion to stay within thermal envelope",
                    "Deploy: Achieve 60 FPS on a constrained-power testbed without thermal throttling",
                ],
            },
            "TINY": {
                "role": "Neural Hearable Lead",
                "goal": "Real-time speech noise isolation in <10ms using <1mW of power.",
                "nemesis": "SRAM Capacity Wall (Every Byte Counts)",
                "color": COLORS['GreenLine'],
                "bg": COLORS['GreenL'],
                "persona": "The Hardware Lead",
                "quote": (
                    "We have 256KB of on-chip SRAM. Every weight byte you eliminate is "
                    "audio buffer we gain. This is not about elegance. This is a war for bytes."
                ),
                "milestones": [
                    "Foundations: Count every byte in the DS-CNN KWS model using the Iron Law",
                    "Build: Implement depthwise separable convolutions in TinyTorch",
                    "Optimize: Achieve 4× compression via magnitude pruning + INT8 quantization",
                    "Deploy: Fit the full inference pipeline within 256KB on a physical MCU",
                ],
            },
        }

        if kat3_track_radio.value is None:
            return mo.vstack([
                _header,
                kat3_track_radio,
                mo.md("_Select a track above to receive your mission briefing._"),
            ])

        # Side-effect: persist choice to ledger
        ledger.save(track=kat3_track_radio.value, chapter=0)

        _t = _tracks[kat3_track_radio.value]

        _milestone_items = "".join([
            f"""<li style="margin:6px 0; padding:10px 12px; background:white; border-radius:6px;
                border-left:3px solid {_t['color']}; font-size:0.9rem;">{m}</li>"""
            for m in _t["milestones"]
        ])

        _mission_card_html = f"""
        <div style="border:2px solid {_t['color']}; border-radius:12px; padding:1.5rem;
                    background:{_t['bg']}; margin:1rem 0;">
          <div style="font-size:0.7rem; font-weight:700; text-transform:uppercase;
                      color:{_t['color']}; letter-spacing:0.12em;">
            🎖️ MISSION GRANTED
          </div>
          <div style="font-size:1.4rem; font-weight:800; color:{_t['color']}; margin:0.4rem 0;">
            {_t['role']}
          </div>
          <div style="margin:0.4rem 0; font-size:0.95rem;">
            <strong>North Star:</strong> {_t['goal']}
          </div>
          <div style="margin:0.4rem 0; font-size:0.95rem;">
            <strong style="color:{_t['color']};">Arch Nemesis:</strong> {_t['nemesis']}
          </div>
          <div style="margin-top:1.2rem; font-weight:700; font-size:0.9rem;
                      text-transform:uppercase; letter-spacing:0.05em; color:{_t['color']};">
            The 4-Chapter Arc:
          </div>
          <ol style="padding-left:1.5rem; margin:0.5rem 0; list-style:none;">
            {_milestone_items}
          </ol>
        </div>
        """

        _commit_banner = mo.Html(f"""
        <div style="padding:16px; background:{_t['color']}; color:white; border-radius:10px;
                    text-align:center; font-weight:700; margin-top:1rem; font-size:1rem;">
          ✅ Design Ledger Initialized — Track: {_t['role']}<br/>
          <span style="font-weight:400; font-size:0.88rem; opacity:0.9;">
            Proceed to Lab 01: ML Introduction
          </span>
        </div>
        """)

        return mo.vstack([
            _header,
            kat3_track_radio,
            mo.md("---"),
            StakeholderMessage(_t['persona'], _t['quote'], _t['color']),
            mo.Html(_mission_card_html),
            MathPeek(
                r"\text{Arch Nemesis} \xrightarrow{\text{each chapter}} \text{Harder}",
                {
                    "North Star": _t['goal'],
                    "Binding Constraint": _t['nemesis'],
                    "Pattern": "Each lab adds one degree of freedom — and one new dimension of the nemesis",
                }
            ),
            mo.md("---"),
            kat3_reflect,
            _commit_banner,
        ])

    render_kat3()


@app.cell
def __(COLORS, ledger, mo):
    # ── DESIGN LEDGER HUD ────────────────────────────────────────────────────
    # Persistent status bar shown at all times, per the plan's Visual Layout spec.
    _track = ledger.get_track()
    _chapter = ledger._state.current_chapter

    _hud_track_color = {
        "CLOUD": COLORS['BlueLine'],
        "EDGE": COLORS['RedLine'],
        "MOBILE": COLORS['OrangeLine'],
        "TINY": COLORS['GreenLine'],
        "NONE": COLORS['Grey'],
    }.get(_track, COLORS['Grey'])

    _hud_status = "Uninitialized" if _track == "NONE" else f"Active — Chapter {_chapter}"

    mo.Html(f"""
    <div style="display:flex; gap:2rem; align-items:center; padding:10px 20px;
                background:#1a1a2e; border-radius:8px; margin-top:2rem;
                font-family:'SF Mono', monospace; font-size:0.82rem;">
      <div style="color:#888;">🗂️ DESIGN LEDGER</div>
      <div>
        <span style="color:#888;">Track: </span>
        <span style="color:{_hud_track_color}; font-weight:700;">{_track}</span>
      </div>
      <div>
        <span style="color:#888;">Chapter: </span>
        <span style="color:white;">{_chapter}</span>
      </div>
      <div>
        <span style="color:#888;">Status: </span>
        <span style="color:{'#00ff88' if _track != 'NONE' else '#ff6b6b'};">{_hud_status}</span>
      </div>
    </div>
    """)


if __name__ == "__main__":
    app.run()

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
    
    # --- PATHS ---
    try:
        notebook_path = Path(os.path.abspath(__file__))
    except NameError:
        notebook_path = Path(os.getcwd()) / "labs" / "vol1" / "lab_01_ml_intro.py"
    
    project_root = notebook_path.parents[2]
    sys.path.append(str(project_root / "book" / "quarto"))
    sys.path.append(str(project_root))
    
    from mlsys import Engine, Models, Systems, ureg, Q_
    from labs.core.style import COLORS, LAB_CSS, apply_plotly_theme
    from labs.core.components import Card, PredictionLock, MetricRow, ComparisonRow, StakeholderMessage, MathPeek
    from labs.core.state import DesignLedger
    
    ledger = DesignLedger()
    
    return (
        COLORS,
        Card,
        ComparisonRow,
        DesignLedger,
        Engine,
        LAB_CSS,
        MathPeek,
        MetricRow,
        Models,
        PredictionLock,
        Q_,
        StakeholderMessage,
        Systems,
        apply_plotly_theme,
        go,
        ledger,
        mo,
        np,
        os,
        project_root,
        sys,
        ureg,
    )


@app.cell
def __(LAB_CSS, mo):
    # --- GLOBAL PRIMER ---
    mo.vstack([
        LAB_CSS,
        mo.md("# 🚀 Lab 01: The Quantitative Orientation"),
        mo.md(
            r"""
            ### **The End of Software 1.0**
            In traditional software, we write explicit instructions. In Machine Learning, we write procedures that extract logic *from data*. This shift transforms our role from programmers to **System Architects**. We no longer just manage code; we manage a **Dual Mandate**: the statistical uncertainty of learned behavior and the physical constraints of the machine executing it.
            
            This lab is your orientation. In the tradition of **Hennessy & Patterson**, we replace intuition with measurement. You will use these analytical instruments to **witness the physics** of the AI landscape before you claim your specialization track.
            """
        )
    ])


@app.cell
def __(mo):
    # --- NAVIGATION ---
    tabs = mo.ui.tabs({
        "1. THE MAGNITUDE GAP": mo.md(""), 
        "2. THE BITTER LESSON": mo.md(""),
        "3. THE VERIFICATION GAP": mo.md(""),
        "4. CLAIM MISSION": mo.md("")
    })
    tabs
    return (tabs,)


@app.cell
def __(PredictionLock, ledger, mo):
    # --- GLOBAL UI STATE ---
    # Part 1
    p1_val, p1_lock_ui = PredictionLock(1, "By what factor (ratio) does an H100 (Cloud) exceed an ESP32 (TinyML) in peak compute?")
    p1_tier_slider = mo.ui.slider(start=0, stop=3, step=1, value=0, label="Select Target Archetype (Tiny → Cloud)")
    
    # Part 2
    p2_val, p2_lock_ui = PredictionLock(2, "Based on historical data, which axis has delivered more 'Accuracy-per-Dollar': Human Tuning or Machine Scaling?")
    p2_time_slider = mo.ui.slider(start=2012, stop=2024, step=1, value=2012, label="Year of Analysis")
    
    # Part 3
    p3_val, p3_lock_ui = PredictionLock(3, "Given the input space of a standard vision model, can we achieve 1% coverage through testing alone?")
    p3_img_res = mo.ui.number(start=32, stop=1024, step=32, value=224, label="Input Resolution (Width/Height)")
    p3_test_rate = mo.ui.number(start=1, stop=10000, value=100, label="Test Rate (Samples/Second)")
    
    # Part 4 - Initialize from Ledger
    initial_track = ledger.get_track()
    if initial_track == "NONE": initial_track = None
    
    track_selector = mo.ui.radio(
        options={
            "☁️ Cloud Titan (LLM Serving)": "CLOUD",
            "🤖 Edge Guardian (AV Systems)": "EDGE",
            "🕶️ Mobile Nomad (AR Glasses)": "MOBILE",
            "👂 Tiny Pioneer (Neural Hearing)": "TINY"
        },
        value=initial_track,
        label="Select your Career Specialization Track"
    )
    
    return (
        p1_lock_ui,
        p1_tier_slider,
        p1_val,
        p2_lock_ui,
        p2_time_slider,
        p2_val,
        p3_img_res,
        p3_lock_ui,
        p3_test_rate,
        p3_val,
        track_selector,
    )


@app.cell
def __(
    Card,
    ComparisonRow,
    MathPeek,
    Systems,
    apply_plotly_theme,
    go,
    mo,
    np,
    p1_lock_ui,
    p1_tier_slider,
    p1_val,
    tabs,
):
    # --- TAB 1: THE MAGNITUDE GAP ---
    def render_tab_1():
        mo.stop(tabs.value != "1. THE MAGNITUDE GAP")
        
        _header = mo.md(r"""
            ## Part 1: The Magnitude Gap (D·A·M Taxonomy)
            
            Every ML system is a three-way interaction between the **Algorithm**, the **Data**, and the **Machine**. We call this the **AI Triad**. 
            
            The central challenge of AI Engineering is that the landscape is not flat. It spans **nine orders of magnitude**. The gap between an H100 GPU and an ESP32 microcontroller is larger than the gap between a domestic house and the Burj Khalifa. 
            
            **The Architect's Invariant:** You cannot simply "shrink" a model across regimes. A model designed for the 80GB VRAM of a Cloud node requires a fundamental physical redesign to fit the 512KB of a TinyML node. As you move the slider, observe the **Ratios**—these multipliers define the boundary of what is possible in your track.
        """)
        
        if p1_val.value == "": return mo.vstack([_header, p1_lock_ui])
        
        baseline = Systems.Tiny
        _tiers = [Systems.Tiny, Systems.Mobile, Systems.Edge, Systems.Cloud]
        selected = _tiers[p1_tier_slider.value]
        
        # Plot
        fig = go.Figure(go.Bar(
            x=['RAM', 'Compute', 'Power'],
            y=[np.log10(max(selected.ram.m_as('GB'), 1e-6)),
               np.log10(max(selected.peak_flops.m_as('TFLOPs/s'), 1e-6)),
               np.log10(max(selected.power_budget.m_as('watt'), 1e-6))],
            marker_color="#006395"
        ))
        fig.update_layout(yaxis=dict(range=[-6, 6], title="Log10 Magnitude (Orders)"), height=250)
        
        comp_metrics = mo.Html("".join([
            ComparisonRow("RAM Gap", baseline.ram, selected.ram, "Bytes"),
            ComparisonRow("Compute Gap", baseline.peak_flops, selected.peak_flops, "TFLOPS"),
            ComparisonRow("Power Gap", baseline.power_budget, selected.power_budget, "Watts")
        ]))
        
        return mo.vstack([
            _header,
            mo.md("#### **Instrument: The Scale Comparator**"),
            p1_tier_slider,
            mo.hstack([
                Card(f"Archetype: {selected.name}", mo.as_html(apply_plotly_theme(fig))),
                Card("Scaling Ratios (vs. Tiny)", comp_metrics)
            ], widths=[2, 1]),
            MathPeek("Ratio = \text{Target} / \text{Baseline}", {"Tiny Baseline": baseline.name, "Unit": "Log10 Scale"}),
            mo.ui.text_area(label="REFLECT: Based on these ratios, why is 'Cloud-first' development a dangerous strategy for Edge engineers?")
        ])

    render_tab_1()


@app.cell
def __(
    COLORS,
    Card,
    MathPeek,
    MetricRow,
    apply_plotly_theme,
    go,
    mo,
    p2_lock_ui,
    p2_time_slider,
    p2_val,
    tabs,
):
    # --- TAB 2: THE BITTER LESSON ---
    def render_tab_2():
        mo.stop(tabs.value != "2. THE BITTER LESSON")
        
        _header = mo.md(r"""
            ## Part 2: The Bitter Lesson
            
            In 2019, Rich Sutton formalized an observation that defines the modern era: **General methods that leverage computation consistently outperform approaches that encode human expertise.**
            
            For 50 years, AI tried to use **Rules (Software 1.0)**. In 2012, with the arrival of AlexNet and the GPU, the industry hit a "Take-off" point where **Learning (Software 2.0)** began to scale exponentially.
            
            **The Architect's Invariant:** Human algorithms are a depreciating asset; Machine Scale is a compound interest asset. As you slide the timeline, watch the **Gap** between hand-coded logic and learned patterns. Your job as an architect is to provide the "Surface Area" (compute and data) for this learning to occur.
        """)
        
        if p2_val.value == "": return mo.vstack([_header, p2_lock_ui])
        
        # Historical Benchmarks
        history = [
            {"year": 2012, "name": "AlexNet", "flops": 1.2e18, "acc": 84.7},
            {"year": 2015, "name": "ResNet-50", "flops": 5.0e19, "acc": 92.4},
            {"year": 2018, "name": "BERT-Large", "flops": 3.0e21, "acc": 95.0},
            {"year": 2020, "name": "GPT-3", "flops": 3.1e23, "acc": 98.0},
            {"year": 2023, "name": "GPT-4", "flops": 2.0e25, "acc": 99.0}
        ]
        visible = [p for p in history if p['year'] <= p2_time_slider.value]
        latest = visible[-1]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[p['flops'] for p in history], y=[p['acc'] for p in history], mode='lines', line=dict(color="#bdc3c7", dash='dot'), name="Scaling Law"))
        fig.add_trace(go.Scatter(x=[p['flops'] for p in visible], y=[p['acc'] for p in visible], mode='markers+text', text=[p['name'] for p in visible], marker=dict(size=12, color="#CB202D")))
        fig.update_layout(xaxis_type="log", xaxis_title="Compute (FLOPs)", yaxis_title="Capability", height=250)
        
        _stats = mo.Html("".join([
            MetricRow("Leading Model", latest['name']),
            MetricRow("Compute Scale", f"{latest['flops']:.1e} FLOPs"),
            MetricRow("Historical Phase", "The Scaling Take-off" if latest['year'] >= 2012 else "The Rule Era")
        ]))
        
        return mo.vstack([
            _header,
            mo.md("#### **Instrument: The Evolution Timeline**"),
            p2_time_slider,
            mo.hstack([
                Card("Software 2.0 Trajectory", mo.as_html(apply_plotly_theme(fig))),
                Card("System State", _stats)
            ], widths=[2, 1]),
            MathPeek("Accuracy \propto \text{Compute}^k", {"k": "Scaling Exponent", "Baseline": "2012 AlexNet"}),
            mo.ui.text_area(label="REFLECT: If scale is the ultimate lever, what becomes the primary bottleneck for the AI Architect?")
        ])

    render_tab_2()


@app.cell
def __(
    Card,
    MetricRow,
    mo,
    np,
    p3_img_res,
    p3_lock_ui,
    p3_test_rate,
    p3_val,
    tabs,
):
    # --- TAB 3: THE VERIFICATION GAP ---
    def render_tab_3():
        mo.stop(tabs.value != "3. THE VERIFICATION GAP")
        
        _header = mo.md(r"""
            ## Part 3: The Verification Gap
            
            In traditional software, we can test every branch of logic. In ML, the input space is **high-dimensional**. A single image contains so many possible configurations that it is mathematically impossible to sample the space meaningfully.
            
            **The Architect's Invariant:** We trade *guaranteed correctness* for *statistical reliability*. If you cannot test the entire space, you must engineer for **Observability**. This is the physical justification for MLOps and production monitoring.
            
            **Mission:** Use the calculator to see how long it would take to cover just **1%** of your model's input space using brute-force testing.
        """)
        
        if p3_val.value == "": return mo.vstack([_header, p3_lock_ui])
        
        num_pixels = p3_img_res.value * p3_img_res.value * 3
        log_space = num_pixels * np.log10(256)
        samples_per_year = p3_test_rate.value * 3600 * 24 * 365
        
        _results = mo.Html("".join([
            MetricRow("Input State Space", f"10^{log_space:,.0f}", "Possible RGB configs"),
            MetricRow("Test Capacity", f"{samples_per_year:,.0e} / yr", f"At {p3_test_rate.value} s/s"),
            MetricRow("Coverage Reality", "0.000...00%", "Essentially Zero")
        ]))
        
        return mo.vstack([
            _header,
            mo.md("#### **Instrument: The Verification Auditor**"),
            mo.hstack([p3_img_res, p3_test_rate]),
            mo.hstack([
                Card("The Untestable Space", mo.md(f"A **{p3_img_res.value}x{p3_img_res.value}** image is a vector in a space so vast that no test suite can ever characterize its behavior. You are building a system that will inevitably encounter 'Invisible' states.")),
                Card("Audit Results", _results)
            ], widths=[1, 1]),
            mo.ui.text_area(label="REFLECT: Given this gap, identify the primary risk of deploying an AI system without a monitoring feedback loop.")
        ])

    render_tab_3()


@app.cell
def __(COLORS, Card, StakeholderMessage, ledger, mo, tabs, track_selector):
    # --- TAB 4: CLAIM MISSION ---
    def render_tab_4():
        mo.stop(tabs.value != "4. CLAIM MISSION")
        
        _header = mo.md(r"""
            ## Part 4: Choose Your physical Regime
            
            You have witnessed the foundations: the asymmetry of hardware, the power of scaling, and the impossibility of exhaustive testing. Now, you must choose your path.
            
            In the next 15 labs, you will refine a single, high-stakes system. Your track defines your **Fixed North Star Mission**. While the engineering principles remain universal, the 'Arch Nemesis' constraint you fight in every chapter will differ.
        """)
        
        _data = {
            "CLOUD": ("LLM Architect", "Maximize serving for Llama-3-70B on one H100.", COLORS['BlueLine'], "CFO: We are burning $10k/day. If utilization isn't 80%, you're fired."),
            "EDGE": ("AV Systems Lead", "Maintain 10ms safety-critical vision-to-brake loop.", COLORS['RedLine'], "Safety Lead: A 5ms jitter spike caused a phantom brake event. Fix it."),
            "MOBILE": ("AR Glasses Dev", "Run 60FPS translation overlay under 2W thermal cap.", COLORS['OrangeLine'], "UX Designer: User reports the frames are 'uncomfortably warm' after 2 minutes."),
            "TINY": ("Neural Hearable Lead", "Real-time speech isolation in <10ms under 1mW.", COLORS['GreenLine'], "Hardware Lead: We have 256KB SRAM. Every weight bit is a liability.")
        }
        
        _mission_ui = mo.md("_Please choose a specialization path above to initialize your ledger._")
        if track_selector.value:
            # Side-effect: Save choice to ledger
            ledger.save(track=track_selector.value, chapter=1)
            
            name, mission, color, quote = _data[track_selector.value]
            _mission_ui = mo.vstack([
                StakeholderMessage(name, quote, color),
                Card("Design Ledger: Mission Initialized", f"**Persona:** {name}<br/>**North Star:** {mission}<br/><br/>✅ **Quantitative intuition calibrated.**<br/>Proceed to **Lab 02: ML Systems**.")
            ])
            
        return mo.vstack([
            _header,
            track_selector,
            mo.md("---"),
            _mission_ui
        ])

    render_tab_4()


if __name__ == "__main__":
    app.run()

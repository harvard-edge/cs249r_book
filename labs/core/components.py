# labs/core/components.py
import marimo as mo
from .style import COLORS, apply_plotly_theme

def Card(title, content):
    return mo.Html(f"""
    <div class="lab-card">
        <h3>{title}</h3>
        <div style="flex-grow: 1;">{content}</div>
    </div>
    """)

def MetricRow(label, value, sub_value=""):
    return f"""
    <div class="metric-row">
        <div class="metric-label">
            {label}
            {f"<br/><span style='font-size:0.7rem; font-weight:normal;'>{sub_value}</span>" if sub_value else ""}
        </div>
        <span class="metric-value">{value}</span>
    </div>
    """

def ComparisonRow(label, current, target, unit=""):
    """Calculates the ratio between two values for quantitative intuition."""
    # Convert Pint quantities to magnitude if needed
    val_c = current.m if hasattr(current, 'm') else current
    val_t = target.m if hasattr(target, 'm') else target
    
    ratio = val_t / val_c if val_c > 0 else 0
    ratio_str = f"{ratio:,.0f}x" if ratio >= 1 else f"{ratio:.4f}x"
    
    return f"""
    <div class="metric-row" style="border-bottom: 1px dashed #eee;">
        <span class="metric-label">{label} ({unit})</span>
        <span class="metric-value" style="color: {'#CB202D' if ratio > 1000 else '#006395'}">{ratio_str}</span>
    </div>
    """

def PredictionLock(task_id, question):
    prediction = mo.ui.text_area(
        label=f"PREDICT: {question}",
        placeholder="Type your quantitative hypothesis...",
        full_width=True
    )
    ui = mo.vstack([
        prediction,
        mo.Html(f"""
        <div class="prediction-box">
            <span style="font-weight:700;">⚠️ TASK {task_id} LOCKED</span><br/>
            Analyze the textbook math and enter a prediction to unlock the instruments.
        </div>
        """)
    ])
    return prediction, ui

def StakeholderMessage(persona, quote, color=COLORS['BlueLine']):
    return mo.Html(f"""
    <div class="stakeholder-card" style="border-left-color: {color};">
        <div style="font-size: 0.8rem; font-weight: 700; color: #7f8c8d; text-transform: uppercase;">Incoming Message: {persona}</div>
        <div style="font-style: italic; margin-top: 8px; color: #2c3e50; font-size: 1.1rem;">"{quote}"</div>
    </div>
    """)

def RooflineVisualizer(system, profile):
    import plotly.graph_objects as go
    import numpy as np
    
    peak_gflops = system.peak_flops.to('GFLOPs/s').magnitude
    bw_gbs = system.memory_bw.to('GB/s').magnitude
    
    x_range = np.logspace(-1, 4, 100)
    y_roof = np.minimum(x_range * bw_gbs, peak_gflops)
    
    fig = go.Figure()
    # HW Limit
    fig.add_trace(go.Scatter(
        x=x_range, y=y_roof, mode='lines', name='Hardware Limit',
        line=dict(color=COLORS['Grey'], width=3), fill='tozeroy', 
        fillcolor='rgba(189, 195, 199, 0.1)'
    ))
    # Model Point
    model_ai = profile.arithmetic_intensity.magnitude
    model_perf = min(model_ai * bw_gbs, peak_gflops)
    fig.add_trace(go.Scatter(
        x=[model_ai], y=[model_perf], mode='markers', name='Design Point',
        marker=dict(color=COLORS['BlueLine'], size=16, symbol='diamond', line=dict(color='white', width=2))
    ))
    
    fig.update_layout(
        xaxis=dict(title="Arithmetic Intensity (Ops/Byte)", type='log'),
        yaxis=dict(title="Throughput (GFLOPS)", type='log'),
        height=300, showlegend=False
    )
    
    return apply_plotly_theme(fig)

def LatencyWaterfall(profile):
    import plotly.graph_objects as go
    
    t_comp = profile.latency_compute.to('ms').magnitude
    t_mem = profile.latency_memory.to('ms').magnitude
    t_ovh = profile.latency_overhead.to('ms').magnitude
    
    fig = go.Figure(go.Bar(
        x=['Compute', 'Memory', 'Overhead'],
        y=[t_comp, t_mem, t_ovh],
        marker_color=[COLORS['BlueLine'], COLORS['RedLine'], COLORS['OrangeLine']],
        width=0.5
    ))
    
    fig.update_layout(
        height=300, yaxis=dict(title="Milliseconds (ms)"),
        margin=dict(t=20)
    )
    return apply_plotly_theme(fig)

def MathPeek(formula, variables):
    """A small toggle to reveal the underlying physics."""
    rows = "".join([f"<li><b>{k}:</b> {v}</li>" for k, v in variables.items()])
    return mo.accordion({
        "📐 View the Invariant": mo.md(f"""
        **Formula:** `{formula}`
        
        **Components:**
        <ul>{rows}</ul>
        """)
    })

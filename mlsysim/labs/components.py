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

def MapDashboard(active_grid, active_eval, comparison_evals):
    """
    Renders the V3 Map + Donut + Comparison Strip dashboard for Sustainability.
    
    Args:
        active_grid (GridProfile): The currently selected grid.
        active_eval (SystemEvaluation): The evaluation result for the active grid.
        comparison_evals (list[dict]): List of dicts with 'grid' and 'carbon_kg' for the strip.
    """
    import plotly.graph_objects as go
    
    # 1. Map
    fig_map = go.Figure()
    
    for comp in comparison_evals:
        grid = comp['grid']
        is_active = grid.name == active_grid.name
        
        color = COLORS['GreenLine'] if grid.renewable_pct and grid.renewable_pct > 50 else COLORS['OrangeLine']
        if not is_active:
            color = "#cbd5e1"
            
        fig_map.add_trace(go.Scattergeo(
            lon=[grid.lon or 0],
            lat=[grid.lat or 0],
            text=[f"{grid.name}<br>{grid.carbon_intensity_g_kwh} g/kWh"],
            marker=dict(
                size=16 if is_active else 10,
                color=color,
                line=dict(width=2, color='white')
            ),
            name=grid.name
        ))
        
    fig_map.update_layout(
        geo=dict(
            projection_type="natural earth",
            showland=True,
            landcolor="#e2e8f0",
            showocean=True,
            oceancolor="#f8fafc",
            showcountries=True,
            countrycolor="#ffffff",
            coastlinecolor="#ffffff"
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=250,
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    
    # 2. Donut Chart
    pct = active_grid.renewable_pct or 0.0
    fig_donut = go.Figure(go.Pie(
        values=[pct, 100 - pct],
        labels=['Renewable', 'Other'],
        hole=.75,
        marker=dict(colors=[COLORS['GreenLine'] if pct > 50 else COLORS['OrangeLine'], '#f1f5f9']),
        textinfo='none',
        hoverinfo='label+percent'
    ))
    fig_donut.update_layout(
        showlegend=False, 
        height=140, width=140,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        annotations=[dict(text=f"<b>{pct:.0f}%</b><br><span style='font-size:10px;color:#64748b;'>renewable</span>", x=0.5, y=0.5, font_size=18, showarrow=False, font_color=COLORS['Text'])]
    )
    
    # 3. Metrics HTML
    carbon_t = active_eval.macro.metrics.get('carbon_footprint', 0)
    energy = active_eval.macro.metrics.get('energy_cost', 0) # Just a placeholder metric
    
    metrics_html = f"""
    <div style="display:flex; flex-direction:column; gap:8px; justify-content:center; padding-left:16px;">
        <div style="display:flex; justify-content:space-between; width:150px; font-size:0.85rem;">
            <span style="color:#64748b; font-weight:600;">CO₂</span>
            <span style="font-weight:800; color:{COLORS['GreenLine'] if pct>50 else COLORS['OrangeLine']};">{carbon_t:,.1f} t</span>
        </div>
        <div style="display:flex; justify-content:space-between; width:150px; font-size:0.85rem;">
            <span style="color:#64748b; font-weight:600;">PUE</span>
            <span style="font-weight:800; color:{COLORS['Text']};">{active_grid.pue:.2f}</span>
        </div>
        <div style="display:flex; justify-content:space-between; width:150px; font-size:0.85rem;">
            <span style="color:#64748b; font-weight:600;">Grid</span>
            <span style="font-weight:800; color:{COLORS['Text']};">{active_grid.carbon_intensity_g_kwh} <span style="font-size:0.6rem;font-weight:normal;">g/kWh</span></span>
        </div>
    </div>
    """
    
    # 4. Strip HTML
    strip_items = []
    for comp in comparison_evals:
        grid = comp['grid']
        val_t = comp['carbon_kg'] / 1000.0
        is_active = grid.name == active_grid.name
        
        bg = "#ffffff" if is_active else "#f8fafc"
        border = f"2px solid {COLORS['GreenLine'] if pct>50 else COLORS['OrangeLine']}" if is_active else "1px solid #e2e8f0"
        
        strip_items.append(f"""
        <div style="flex:1; background:{bg}; border:{border}; border-radius:8px; padding:10px; text-align:center;">
            <div style="font-size:0.75rem; font-weight:700; color:{COLORS['TextMuted']};">{grid.name.split(' ')[0]}</div>
            <div style="font-size:1.1rem; font-weight:800; color:{COLORS['Text']};">{val_t:,.1f} t</div>
        </div>
        """)
        
    strip_html = f'<div style="display:flex; gap:12px; margin-top:20px;">{"".join(strip_items)}</div>'
    
    # Assembly
    return mo.vstack([
        mo.Html("""<div style="font-size:0.8rem; font-weight:700; color:#64748b; margin-bottom:8px;">Geographic Context</div>"""),
        mo.as_html(fig_map),
        mo.Html(f"""
        <div style="background:white; border:1px solid #e2e8f0; border-radius:12px; padding:20px; margin-top:16px;">
            <div style="display:flex; align-items:center;">
                {mo.as_html(fig_donut).text}
                {metrics_html}
            </div>
            {strip_html}
        </div>
        """)
    ])

def DecisionLog(placeholder="I chose this configuration because..."):
    """The Commit phase: requires a student to justify their setup."""
    text_input = mo.ui.text_area(label="Justify your architectural choice:", placeholder=placeholder, full_width=True)
    ui = mo.vstack([
        mo.Html("""<div style="font-size:0.9rem; font-weight:700; color:#1e293b; margin:16px 0 8px 0;">Decision Log</div>"""),
        text_input
    ])
    return text_input, ui

def HardwareTetrisDashboard(eval_obj):
    """
    Renders the V3 Dashboard for Compute Infrastructure (Hardware Tetris).
    Visualizes how the workload fits into the selected hardware footprint.
    """
    import plotly.graph_objects as go
    
    perf = eval_obj.performance.metrics
    feas = eval_obj.feasibility.metrics
    
    fig = go.Figure()
    
    # 1. MFU Gauge
    mfu_val = perf.get('mfu', 0)
    if mfu_val == 0 and 'fleet_throughput' in perf:
        # It's a distributed result without pure MFU in top level metrics, approximate
        mfu_val = 0.52 * perf.get('fleet_throughput', 0) / 10000.0 # Just a placeholder
        
    if mfu_val <= 1.0: mfu_val *= 100.0 # Convert to percent
        
    fig.add_trace(go.Indicator(
        mode = "gauge+number",
        value = mfu_val,
        title = {'text': "Model FLOPs Utilization", 'font': {'size': 14}},
        domain = {'x': [0, 0.45], 'y': [0, 1]},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': COLORS['GreenLine'] if mfu_val > 40 else COLORS['OrangeLine']},
            'steps': [
                {'range': [0, 20], 'color': 'rgba(203,32,45,0.1)'},
                {'range': [20, 40], 'color': 'rgba(245,158,11,0.1)'},
                {'range': [40, 100], 'color': 'rgba(16,185,129,0.1)'}
            ]
        }
    ))
    
    # 2. Memory Footprint Gauge
    mem_used = feas.get('memory_used_gb', 0)
    # Estimate total available if not provided directly
    mem_total = mem_used * 1.5 if mem_used > 0 else 80 
    
    fig.add_trace(go.Indicator(
        mode = "gauge+number",
        value = mem_used,
        title = {'text': "Memory Footprint (GB)", 'font': {'size': 14}},
        domain = {'x': [0.55, 1], 'y': [0, 1]},
        gauge = {
            'axis': {'range': [None, mem_total * 1.2]},
            'bar': {'color': COLORS['RedLine'] if mem_used > mem_total else COLORS['BlueLine']},
            'steps': [
                {'range': [0, mem_total], 'color': 'rgba(16,185,129,0.1)'},
                {'range': [mem_total, mem_total * 1.2], 'color': 'rgba(203,32,45,0.1)'}
            ],
            'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': mem_total}
        }
    ))
    
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20))
    apply_plotly_theme(fig)
    
    return mo.vstack([
        mo.Html("""<div style="font-size:0.8rem; font-weight:700; color:#64748b; margin-bottom:8px;">Systems Telemetry</div>"""),
        mo.as_html(fig)
    ])

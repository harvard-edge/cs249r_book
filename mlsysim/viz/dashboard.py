# mlsysim/viz/dashboard.py
"""
MLSySim Universal Dashboard Components
======================================
Enforces the 4-Zone "Cockpit" layout for all Marimo labs.
Includes high-fidelity CSS for pedagogical guidance.
"""

import marimo as mo
import plotly.graph_objects as go
from typing import Dict, List, Tuple

def command_header(title: str, subtitle: str, persona_name: str, scale: str, constraints: Dict[str, bool]):
    """Zone 1: Identity & Status."""
    badges_html = ""
    for label, is_met in constraints.items():
        color = "#008f45" if is_met else "#cb202d"
        icon = "✅" if is_met else "❌"
        badges_html += f'<span style="background-color: {color}; color: white; padding: 4px 10px; border-radius: 12px; font-size: 0.8em; margin-right: 10px; font-weight: bold;">{icon} {label}</span>'

    return mo.md(
        f"""
        <div style="background: linear-gradient(135deg, #1a202c 0%, #2d3748 100%); padding: 25px; border-radius: 12px; color: white; margin-bottom: 20px; box-shadow: 0 4px 15px rgba(0,0,0,0.2);">
            <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                <div>
                    <h1 style="margin: 0; font-size: 2.2em; color: #e2e8f0; font-weight: 800;">{title}</h1>
                    <p style="margin: 5px 0 15px 0; color: #a0aec0; font-size: 1.1em;">{subtitle}</p>
                    <div style="display: flex; flex-wrap: wrap;">{badges_html}</div>
                </div>
                <div style="text-align: right; border-left: 2px solid rgba(255,255,255,0.1); padding-left: 20px;">
                    <div style="font-weight: bold; font-size: 1.2em; color: #63b3ed;">👤 {persona_name}</div>
                    <div style="font-size: 0.9em; color: #cbd5e0; margin-top: 5px;">📊 Scale: {scale}</div>
                </div>
            </div>
        </div>
        """
    )

def pro_note(title: str, body: str):
    """A styled callout box for pedagogical notes/guidance."""
    return mo.md(
        f"""
        <div style="background-color: #ebf8ff; border-left: 6px solid #3182ce; padding: 15px; border-radius: 4px; margin: 15px 0;">
            <div style="font-weight: 800; color: #2c5282; margin-bottom: 5px; text-transform: uppercase; font-size: 0.85em;">💡 Professor's Note: {title}</div>
            <div style="color: #2a4365; line-height: 1.5;">{body}</div>
        </div>
        """
    )

def lever_panel(content):
    """Zone 2: Control Panel container."""
    return mo.md(
        f"""
        <div style="background: #f7fafc; padding: 20px; border-radius: 12px; border: 1px solid #e2e8f0; box-shadow: inset 0 2px 4px rgba(0,0,0,0.05);">
            <div style="font-weight: 800; color: #4a5568; margin-bottom: 15px; text-transform: uppercase; font-size: 0.8em; letter-spacing: 1px;">🎛️ Control Levers</div>
            {mo.as_html(content)}
        </div>
        """
    )

def telemetry_panel(content, color="#3182ce"):
    """Zone 3: Instrument Cluster container."""
    return mo.md(
        f"""
        <div style="background: white; padding: 25px; border-radius: 12px; border-top: 6px solid {color}; box-shadow: 0 4px 12px rgba(0,0,0,0.08); height: 100%;">
            <div style="font-weight: 800; color: #4a5568; margin-bottom: 15px; text-transform: uppercase; font-size: 0.8em; letter-spacing: 1px;">📡 Live Telemetry</div>
            {mo.as_html(content)}
        </div>
        """
    )

def audit_panel(content):
    """Zone 4: Log & Justification container."""
    return mo.md(
        f"""
        <div style="background: #fffaf0; padding: 20px; border-radius: 12px; border: 1px solid #feebc8; margin-top: 20px;">
            <div style="font-weight: 800; color: #c05621; margin-bottom: 10px; text-transform: uppercase; font-size: 0.8em; letter-spacing: 1px;">📝 Audit Trail & Rationale</div>
            {mo.as_html(content)}
        </div>
        """
    )

def metric_card(label: str, value: str, trend: str, detail: str, color: str = "#3182ce"):
    """High-density scorecard card."""
    return mo.md(
        f"""
        <div style="background: #fdfdfd; padding: 15px; border-radius: 8px; border: 1px solid #edf2f7; height: 100%;">
            <div style="font-size: 0.7em; color: #718096; font-weight: 800; text-transform: uppercase;">{label}</div>
            <div style="font-size: 1.6em; font-weight: 900; color: #2d3748; margin: 5px 0;">{value}</div>
            <div style="font-size: 0.75em; color: {color}; font-weight: bold;">{trend} <span style="color: #a0aec0; font-weight: normal;">| {detail}</span></div>
        </div>
        """
    )

def layout_cockpit(levers_ui, telemetry_ui, audit_ui):
    """Universal Grid Layout."""
    return mo.vstack([
        mo.hstack([levers_ui, telemetry_ui], widths=[4, 6], gap=2),
        audit_ui
    ], gap=1)

def pareto_plot(x_val: float, y_val: float, x_label: str, y_label: str, title: str, 
                pareto_x: List[float], pareto_y: List[float]):
    """Standard Pareto plotter."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pareto_x, y=pareto_y, mode='lines', name='Optimum', line=dict(color='rgba(160, 174, 192, 0.5)', width=3, dash='dash')))
    fig.add_trace(go.Scatter(x=[x_val], y=[y_val], mode='markers', name='State', marker=dict(color='#e53e3e', size=16, symbol='cross', line=dict(width=2, color='white'))))
    fig.update_layout(title=title, xaxis_title=x_label, yaxis_title=y_label, template="plotly_white", margin=dict(l=40, r=40, t=40, b=40), height=350)
    return fig

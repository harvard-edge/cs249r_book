# mlsysim/viz/dashboard.py
"""
MLSySim Universal Dashboard Components
======================================
Enforces the 4-Zone "Cockpit" layout for all Marimo labs:
1. Command Header (Identity & Constraints)
2. Engineering Levers (Inputs)
3. Telemetry Center (Scorecards & Pareto Plots)
4. Audit Trail (Logs & Justification)
"""

import marimo as mo
import plotly.graph_objects as go
from typing import Dict, List, Tuple

def command_header(title: str, subtitle: str, persona_name: str, scale: str, constraints: Dict[str, bool]):
    """
    Renders Zone 1: The Command Header.
    Includes dynamic status badges for physical/economic constraints.
    """
    # Build constraint badges
    badges_html = ""
    for label, is_met in constraints.items():
        color = "#008f45" if is_met else "#cb202d"
        icon = "✅" if is_met else "❌"
        badges_html += f'<span style="background-color: {color}; color: white; padding: 4px 8px; border-radius: 12px; font-size: 0.8em; margin-right: 10px; font-weight: bold;">{icon} {label}</span>'

    return mo.md(
        f"""
        <div style="background: linear-gradient(135deg, #1a202c 0%, #2d3748 100%); padding: 25px; border-radius: 12px; color: white; margin-bottom: 20px; box-shadow: 0 4px 15px rgba(0,0,0,0.2);">
            <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                <div>
                    <h1 style="margin: 0; font-size: 2.2em; color: #e2e8f0;">{title}</h1>
                    <p style="margin: 5px 0 15px 0; color: #a0aec0; font-size: 1.1em;">{subtitle}</p>
                    <div style="display: flex; flex-wrap: wrap;">
                        {badges_html}
                    </div>
                </div>
                <div style="text-align: right; border-left: 2px solid rgba(255,255,255,0.1); padding-left: 20px;">
                    <div style="font-weight: bold; font-size: 1.2em; color: #63b3ed;">👤 {persona_name}</div>
                    <div style="font-size: 0.9em; color: #cbd5e0; margin-top: 5px;">📊 Scale: {scale}</div>
                </div>
            </div>
        </div>
        """
    )

def metric_card(label: str, value: str, trend: str, detail: str, color: str = "#3182ce"):
    """Renders a high-density telemetry card."""
    return mo.md(
        f"""
        <div style="background: white; padding: 15px; border-radius: 10px; border-top: 5px solid {color}; box-shadow: 0 2px 8px rgba(0,0,0,0.08); height: 100%; display: flex; flex-direction: column; justify-content: space-between;">
            <div style="font-size: 0.75em; color: #718096; font-weight: 800; text-transform: uppercase; letter-spacing: 0.5px;">{label}</div>
            <div style="font-size: 1.8em; font-weight: 900; margin: 8px 0; color: #2d3748;">{value}</div>
            <div style="font-size: 0.8em; color: {color}; font-weight: 600;">{trend} <span style="color: #a0aec0; font-weight: normal;">| {detail}</span></div>
        </div>
        """
    )

def telemetry_scorecard(ledger):
    """Renders Zone 3 (Top): The 4-axis System Ledger."""
    return mo.hstack([
        metric_card("Performance", f"{ledger.performance.mfu * 100:.1f}%", "MFU", "Silicon Efficiency", "#3182ce"),
        metric_card("Sustainability", f"{ledger.sustainability.carbon_kg / 1000:,.1f}t", "CO2e", "Fleet Footprint", "#38a169"),
        metric_card("Economics", f"${ledger.economics.tco / 1_000_000:,.1f}M", "TCO", "Annual Cost", "#dd6b20"),
        metric_card("Reliability", f"{ledger.reliability.goodput * 100:.1f}%", "Goodput", "Effective Work", "#e53e3e"),
    ], justify="space-between", gap=1)

def pareto_plot(x_val: float, y_val: float, x_label: str, y_label: str, title: str, 
                pareto_x: List[float], pareto_y: List[float]):
    """Renders Zone 3 (Bottom): The Trade-off Visualizer."""
    fig = go.Figure()
    
    # The Optimal Frontier (Ghost line)
    fig.add_trace(go.Scatter(x=pareto_x, y=pareto_y, mode='lines', 
                             name='Theoretical Optimum', line=dict(color='rgba(160, 174, 192, 0.5)', width=3, dash='dash')))
    
    # The Student's Current State
    fig.add_trace(go.Scatter(x=[x_val], y=[y_val], mode='markers', 
                             name='Your System State', marker=dict(color='#e53e3e', size=16, symbol='cross', line=dict(width=2, color='white'))))

    fig.update_layout(
        title=title, xaxis_title=x_label, yaxis_title=y_label,
        template="plotly_white", margin=dict(l=40, r=40, t=40, b=40),
        height=350, showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def layout_cockpit(levers_ui, telemetry_ui, audit_ui):
    """Assembles the final 4-Zone Cockpit grid."""
    return mo.vstack([
        mo.hstack([
            # ZONE 2: Engineering Levers (30% width)
            mo.vstack([
                mo.md("### 🎛️ Engineering Levers"),
                mo.md("<div style='background: #f7fafc; padding: 15px; border-radius: 8px; border: 1px solid #e2e8f0;'>"),
                levers_ui,
                mo.md("</div>")
            ]),
            # ZONE 3: Telemetry Center (70% width)
            mo.vstack([
                mo.md("### 📡 Telemetry Center"),
                telemetry_ui
            ])
        ], widths=[3, 7], gap=2),
        mo.md("---"),
        # ZONE 4: Audit Trail
        mo.md("### 📝 Audit Trail & Justification"),
        audit_ui
    ], gap=1)

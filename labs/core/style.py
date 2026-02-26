# labs/core/style.py
# The "Gold Standard" Visual Identity for MLSys Labs.

import marimo as mo

COLORS = {
    "BlueLine": "#006395",
    "BlueL": "#D1E6F3",
    "GreenLine": "#008F45",
    "GreenL": "#D4EFDF",
    "RedLine": "#CB202D",
    "RedL": "#F5D2D5",
    "OrangeLine": "#CC5500",
    "OrangeL": "#FFE5CC",
    "Grey": "#bdc3c7",
    "Neutral": "#f8f9fa",
    "Text": "#2c3e50",
    "Border": "#dee2e6"
}

LAB_CSS = mo.Html(f"""
<style>
    :root {{
        --blue: {COLORS['BlueLine']};
        --red: {COLORS['RedLine']};
        --green: {COLORS['GreenLine']};
        --text: {COLORS['Text']};
    }}

    .lab-container {{
        font-family: 'Inter', -apple-system, sans-serif;
        color: var(--text);
        max-width: 900px;
        margin: auto;
    }}

    .lab-card {{
        background-color: white;
        border: 1px solid {COLORS['Border']};
        border-radius: 12px;
        padding: 1.5rem;
        height: 100%;
        box-shadow: 0 4px 12px rgba(0,0,0,0.03);
        display: flex;
        flex-direction: column;
    }}

    .lab-card h3 {{
        margin-top: 0;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #7f8c8d;
        border-bottom: 1px solid #eee;
        padding-bottom: 8px;
        margin-bottom: 1rem;
    }}

    .metric-row {{
        display: flex;
        justify-content: space-between;
        padding: 8px 0;
        border-bottom: 1px solid {COLORS['Neutral']};
    }}

    .metric-label {{ font-weight: 500; color: #7f8c8d; font-size: 0.9rem; }}
    .metric-value {{ font-family: 'SF Mono', monospace; font-weight: 600; color: var(--blue); }}

    .prediction-box {{
        background-color: {COLORS['OrangeL']};
        border-left: 4px solid {COLORS['OrangeLine']};
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }}

    .feasibility-banner {{
        padding: 1rem;
        border-radius: 10px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }}

    .stakeholder-card {{
        background: {COLORS['Neutral']};
        border-radius: 12px;
        padding: 1.5rem;
        border-left: 6px solid var(--blue);
        margin-top: 1rem;
    }}
</style>
""")

def apply_plotly_theme(fig):
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font_family="Inter, sans-serif",
        font_color=COLORS['Text'],
        margin=dict(l=40, r=20, t=20, b=40),
        xaxis=dict(gridcolor='#f5f5f5'),
        yaxis=dict(gridcolor='#f5f5f5')
    )
    return fig

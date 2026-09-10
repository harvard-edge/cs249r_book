# mlsysim/labs/style.py
# MLSys Labs — Unified Design System
#
# USAGE:
#   from mlsysim.labs.style import COLORS, LAB_CSS, progress_bar, concept_header
#   Always inject LAB_CSS once at the top of every lab (in the header cell).
#   Use CSS class names from this file — never write inline style= for structural elements.
#
# DESIGN TOKENS (semantic names map to brand colors):
#   --primary     BlueLine  #006395   data, healthy state, interactive
#   --success     GreenLine #008F45   target met, correct answer, success
#   --danger      RedLine   #CB202D   constraint violated, wrong, OOM
#   --warning     OrangeLine #CC5500  caution, prediction lock, secondary
#   --surface-0   #0f172a             dark diagnostic surfaces
#   --surface-1   #1e293b             card backgrounds (dark)
#   --surface-2   #f8fafc             card backgrounds (light)
#   --border      #e2e8f0             standard separator
#   --text-primary  #0f172a           headings
#   --text-secondary #475569          body text
#   --text-muted    #94a3b8           labels, captions

import marimo as mo

# ── PALETTE ──────────────────────────────────────────────────────────────────

COLORS = {
    # Brand primaries
    "BlueLine":    "#006395",
    "BlueL":       "#D1E6F3",
    "BlueLL":      "#EBF4FA",
    "GreenLine":   "#008F45",
    "GreenL":      "#D4EFDF",
    "GreenLL":     "#ECFDF5",
    "RedLine":     "#CB202D",
    "RedL":        "#F5D2D5",
    "RedLL":       "#FEF2F2",
    "OrangeLine":  "#CC5500",
    "OrangeL":     "#FFE5CC",
    "OrangeLL":    "#FFF7ED",
    # Neutrals
    "Grey":        "#bdc3c7",
    "Neutral":     "#f8f9fa",
    "Surface0":    "#0f172a",
    "Surface1":    "#1e293b",
    "Surface2":    "#f8fafc",
    "Border":      "#e2e8f0",
    # Text
    "Text":        "#0f172a",
    "TextSec":     "#475569",
    "TextMuted":   "#94a3b8",
    # Deployment regime accents (matches PROTOCOL)
    "Cloud":       "#6366f1",   # indigo
    "Edge":        "#CB202D",   # red
    "Mobile":      "#CC5500",   # orange
    "Tiny":        "#008F45",   # green
}

# ── GLOBAL CSS ────────────────────────────────────────────────────────────────
# Inject once at the top of every lab. Contains all structural classes.
# Individual cells should use class names, not inline style=.

LAB_CSS = mo.Html(f"""
<style>
/* ── TOKENS ── */
:root {{
    --primary:        {COLORS['BlueLine']};
    --success:        {COLORS['GreenLine']};
    --danger:         {COLORS['RedLine']};
    --warning:        {COLORS['OrangeLine']};
    --surface-0:      {COLORS['Surface0']};
    --surface-1:      {COLORS['Surface1']};
    --surface-2:      {COLORS['Surface2']};
    --border:         {COLORS['Border']};
    --text-primary:   {COLORS['Text']};
    --text-secondary: {COLORS['TextSec']};
    --text-muted:     {COLORS['TextMuted']};
    --radius-sm:      8px;
    --radius-md:      12px;
    --radius-lg:      16px;
    --font-mono:      'SF Mono', 'Fira Code', 'Cascadia Code', monospace;
    --font-sans:      'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    --readable-width:  720px;
    --panel-width:     840px;
}}

/* ── BASE ── */
body, .lab-body {{
    font-family: var(--font-sans);
    color: var(--text-primary);
    line-height: 1.6;
}}

/* ── READING MEASURE ──
   Narrative blocks should read like an academic worksheet, not stretch across
   the whole browser. Plots and controls are left unconstrained. */
.lab-readable,
.concept-block,
.check-block,
.stakeholder-card,
.prediction-lock-preview,
.orientation-complete,
.lab-hud,
marimo-accordion,
marimo-callout-output {{
    display: block !important;
    width: min(var(--readable-width), 100%) !important;
    max-width: min(var(--readable-width), 100%) !important;
    margin-left: auto !important;
    margin-right: auto !important;
}}

/* Published labs should show course-owned navigation and submission affordances,
   not Marimo runtime chrome. */
div[class~="fixed"][class~="bottom-0"][class~="right-0"][class~="z-50"],
div[class~="fixed"][class~="right-0"][class~="top-0"][class~="z-50"],
div[class~="fixed"][class~="right-8"][class~="z-10000"],
div[class~="fixed"][class~="top-0"][class~="z-100"][class~="max-h-screen"],
ol[class~="fixed"][class~="top-0"][class~="z-100"][class~="max-h-screen"] {{
    display: none !important;
}}

/* Keep ordinary Markdown/HTML narrative cells in the worksheet column while
   leaving charts, tables, and live controls free to use wider space. */
.output.block > div:not(:has(svg, canvas, iframe, table, marimo-ui-element, .js-plotly-plot, .plotly)) {{
    width: min(var(--panel-width), 100%) !important;
    max-width: min(var(--panel-width), 100%) !important;
    margin-left: auto !important;
    margin-right: auto !important;
}}

.output.block > div:not(:has(svg, canvas, iframe, table, marimo-ui-element, .js-plotly-plot, .plotly)) p,
.output.block > div:not(:has(svg, canvas, iframe, table, marimo-ui-element, .js-plotly-plot, .plotly)) li {{
    max-width: min(var(--readable-width), 100%) !important;
}}

marimo-tabs,
div[style*="border-left:4px solid"][style*="border-radius:0 10px"],
div[style*="border-left: 4px solid"][style*="border-radius: 0 10px"],
div[style*="border-left:4px solid"][style*="border-radius:0px 10px"],
div[style*="border-left: 4px solid"][style*="border-radius: 0px 10px"] {{
    display: block !important;
    width: min(var(--panel-width), 100%) !important;
    max-width: min(var(--panel-width), 100%) !important;
    margin-left: auto !important;
    margin-right: auto !important;
}}

marimo-tabs {{
    overflow-x: auto !important;
    overflow-y: visible !important;
}}

/* ── LEGACY LAB HEADER MODERNIZATION ──
   Older notebooks used inline dark gradient hero banners. The selector below
   keeps those notebooks release-ready without hand-editing every header cell. */
div[style*="linear-gradient(135deg, #0f172a"],
div[style*="linear-gradient(135deg, rgb(15, 23, 42)"] {{
    background: #ffffff !important;
    color: var(--text-primary) !important;
    border: 1px solid #cbd5e1 !important;
    border-left: 6px solid #A51C30 !important;
    border-radius: 8px !important;
    box-shadow: 0 2px 10px rgba(15, 23, 42, 0.06) !important;
    padding: 1.5rem 1.75rem !important;
    max-width: min(var(--panel-width), 100%) !important;
    margin-left: auto !important;
    margin-right: auto !important;
}}

div[style*="linear-gradient(135deg, #0f172a"] h1,
div[style*="linear-gradient(135deg, rgb(15, 23, 42)"] h1 {{
    color: var(--text-primary) !important;
    font-size: clamp(2rem, 3vw, 2.65rem) !important;
    letter-spacing: 0 !important;
    margin-bottom: 0.55rem !important;
}}

div[style*="linear-gradient(135deg, #0f172a"] div[style*="font-size: 2.0rem"],
div[style*="linear-gradient(135deg, #0f172a"] div[style*="font-size:2.0rem"],
div[style*="linear-gradient(135deg, rgb(15, 23, 42)"] div[style*="font-size: 2.0rem"],
div[style*="linear-gradient(135deg, rgb(15, 23, 42)"] div[style*="font-size:2.0rem"] {{
    color: var(--text-primary) !important;
    letter-spacing: 0 !important;
}}

div[style*="linear-gradient(135deg, #0f172a"] div[style*="font-weight: 800"],
div[style*="linear-gradient(135deg, rgb(15, 23, 42)"] div[style*="font-weight: 800"],
div[style*="linear-gradient(135deg, #0f172a"] div[style*="color: rgb(241, 245, 249)"],
div[style*="linear-gradient(135deg, rgb(15, 23, 42)"] div[style*="color: rgb(241, 245, 249)"] {{
    color: var(--text-primary) !important;
    letter-spacing: 0 !important;
}}

div[style*="linear-gradient(135deg, #0f172a"] p,
div[style*="linear-gradient(135deg, rgb(15, 23, 42)"] p {{
    color: var(--text-secondary) !important;
    max-width: 760px !important;
}}

div[style*="linear-gradient(135deg, #0f172a"] div[style*="letter-spacing:0.22em"],
div[style*="linear-gradient(135deg, #0f172a"] div[style*="letter-spacing: 0.22em"],
div[style*="linear-gradient(135deg, rgb(15, 23, 42)"] div[style*="letter-spacing:0.22em"],
div[style*="linear-gradient(135deg, rgb(15, 23, 42)"] div[style*="letter-spacing: 0.22em"],
div[style*="linear-gradient(135deg, rgb(15, 23, 42)"] div[style*="letter-spacing: 0.18em"] {{
    color: #64748b !important;
    letter-spacing: 0.12em !important;
}}

div[style*="linear-gradient(135deg, #0f172a"] span,
div[style*="linear-gradient(135deg, rgb(15, 23, 42)"] span {{
    box-shadow: none !important;
}}

div[style*="linear-gradient(135deg, #0f172a"]::after,
div[style*="linear-gradient(135deg, rgb(15, 23, 42)"]::after {{
    content: "MLSysBook Labs v1.0.0 · MLSysIM 0.1.2 · Browser lab";
    display: block;
    margin-top: 16px;
    padding-top: 12px;
    border-top: 1px solid #e2e8f0;
    color: #64748b;
    font-size: 0.76rem;
    font-weight: 650;
    letter-spacing: 0.04em;
}}

/* First-pass width cap for the common inline briefing panels used across the
   generated labs. The selector is intentionally narrow so charts and simulator
   surfaces keep their available width. */
div[style*="border-left: 4px solid"][style*="box-shadow: 0 1px 4px"],
div[style*="border-left:4px solid"][style*="box-shadow:0 1px 4px"],
div[style*="border-left: 4px solid"][style*="background: white"][style*="border-radius: 0 12px 12px 0"],
div[style*="border-left:4px solid"][style*="background:white"][style*="border-radius:0 12px 12px 0"],
div[style*="border-left: 4px solid"][style*="box-shadow"][style*="border-radius: 0px 12px 12px 0px"] {{
    max-width: min(var(--panel-width), 100%) !important;
    margin-left: auto !important;
    margin-right: auto !important;
}}

div[style*="font-style:italic"][style*="line-height:1.65"],
div[style*="font-style: italic"][style*="line-height: 1.65"] {{
    max-width: min(var(--readable-width), 100%);
}}

/* ── CARDS ── */
.lab-card {{
    background: white;
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 1.25rem 1.5rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}}

.lab-card-dark {{
    background: var(--surface-1);
    border: 1px solid #2d3748;
    border-radius: var(--radius-md);
    padding: 1.25rem 1.5rem;
    color: #e2e8f0;
}}

/* ── CONCEPT BLOCK ── */
.concept-block {{
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 1.5rem;
    margin: 1rem 0;
}}

/* ── CHECK BLOCK (MCQ, multiselect) ── */
.check-block {{
    background: white;
    border: 1.5px solid var(--border);
    border-radius: var(--radius-md);
    padding: 1.25rem 1.5rem;
    margin: 0.75rem 0;
}}

.check-correct {{
    background: {COLORS['GreenLL']};
    border-color: {COLORS['GreenLine']};
}}

.check-incorrect {{
    background: {COLORS['RedLL']};
    border-color: {COLORS['RedLine']};
}}

.check-partial {{
    background: {COLORS['OrangeLL']};
    border-color: {COLORS['OrangeLine']};
}}

/* ── PROGRESS INDICATOR ── */
.lab-progress {{
    display: flex;
    align-items: center;
    gap: 8px;
    margin: 0.75rem 0 1.25rem 0;
    font-size: 0.78rem;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.06em;
}}

.lab-progress-dot {{
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: var(--border);
    flex-shrink: 0;
}}

.lab-progress-dot.active {{
    background: var(--primary);
    box-shadow: 0 0 0 3px {COLORS['BlueLL']};
}}

.lab-progress-dot.done {{
    background: var(--success);
}}

.lab-progress-line {{
    flex: 0 0 24px;
    height: 2px;
    background: var(--border);
}}

.lab-progress-line.done {{
    background: var(--success);
}}

/* ── CONCEPT SECTION HEADER ── */
.concept-header {{
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 4px;
    display: flex;
    align-items: center;
    gap: 8px;
}}

.concept-header-line {{
    flex: 1;
    height: 1px;
    background: var(--border);
}}

/* ── CONFIDENCE STRIP ── */
.confidence-strip {{
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 8px 12px;
    background: var(--surface-2);
    border-radius: var(--radius-sm);
    font-size: 0.8rem;
    color: var(--text-muted);
    margin-top: 8px;
}}

/* ── METRIC ROWS ── */
.metric-row {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 0;
    border-bottom: 1px solid var(--surface-2);
}}

.metric-label {{
    font-weight: 500;
    color: var(--text-muted);
    font-size: 0.88rem;
}}

.metric-value {{
    font-family: var(--font-mono);
    font-weight: 700;
    color: var(--primary);
    font-size: 0.95rem;
}}

/* ── CONSTRAINT BADGES ── */
.badge {{
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 700;
    display: inline-flex;
    align-items: center;
    gap: 5px;
    white-space: nowrap;
}}

.badge-ok   {{ background: {COLORS['GreenLL']};  color: {COLORS['GreenLine']}; border: 1px solid {COLORS['GreenL']}; }}
.badge-warn {{ background: {COLORS['OrangeLL']}; color: {COLORS['OrangeLine']}; border: 1px solid {COLORS['OrangeL']}; }}
.badge-fail {{ background: {COLORS['RedLL']};    color: {COLORS['RedLine']};   border: 1px solid {COLORS['RedL']}; }}
.badge-info {{ background: {COLORS['BlueLL']};   color: {COLORS['BlueLine']};  border: 1px solid {COLORS['BlueL']}; }}

/* ── ZONE ANATOMY GRID ── */
.zone-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 14px;
    margin: 1rem 0;
}}

.zone-card {{
    border-radius: var(--radius-md);
    padding: 16px;
    border: 1.5px solid;
    border-top-width: 4px;
}}

.zone-card-title {{
    font-weight: 800;
    font-size: 0.88rem;
    margin-bottom: 6px;
}}

.zone-card-body {{
    font-size: 0.83rem;
    line-height: 1.55;
}}

.zone-1 {{ background: #f0f4ff; border-color: #c7d2fe; border-top-color: #6366f1; }}
.zone-1 .zone-card-title {{ color: #3730a3; }}
.zone-1 .zone-card-body  {{ color: #4338ca; }}

.zone-2 {{ background: {COLORS['GreenLL']}; border-color: #bbf7d0; border-top-color: {COLORS['GreenLine']}; }}
.zone-2 .zone-card-title {{ color: #14532d; }}
.zone-2 .zone-card-body  {{ color: #166534; }}

.zone-3 {{ background: {COLORS['OrangeLL']}; border-color: #fed7aa; border-top-color: #ea580c; }}
.zone-3 .zone-card-title {{ color: #9a3412; }}
.zone-3 .zone-card-body  {{ color: #7c2d12; }}

.zone-4 {{ background: #fffbeb; border-color: #fde68a; border-top-color: #d97706; }}
.zone-4 .zone-card-title {{ color: #92400e; }}
.zone-4 .zone-card-body  {{ color: #78350f; }}

/* ── STAKEHOLDER MESSAGE ── */
.stakeholder-card {{
    border-radius: 0 var(--radius-md) var(--radius-md) 0;
    padding: 16px 22px;
    margin: 12px 0;
    border-left: 4px solid var(--primary);
}}

.stakeholder-byline {{
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 6px;
}}

.stakeholder-quote {{
    font-style: italic;
    font-size: 1.0rem;
    color: var(--text-primary);
    line-height: 1.65;
}}

/* ── PREDICTION LOCK PREVIEW ── */
.prediction-lock-preview {{
    background: var(--surface-1);
    border-radius: var(--radius-md);
    padding: 20px;
    border-left: 4px solid #6366f1;
}}

.prediction-option {{
    background: rgba(99,102,241,0.12);
    border: 1px solid #6366f1;
    border-radius: var(--radius-sm);
    padding: 8px 16px;
    color: #a5b4fc;
    font-size: 0.85rem;
    font-weight: 600;
    cursor: default;
    display: inline-block;
    margin: 4px;
}}

/* ── HUD FOOTER ── */
.lab-hud {{
    display: flex;
    gap: 28px;
    align-items: center;
    padding: 12px 24px;
    background: #ffffff;
    border-radius: var(--radius-md);
    margin-top: 32px;
    font-family: var(--font-mono);
    font-size: 0.8rem;
    border: 1px solid var(--border);
    color: var(--text-secondary);
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    flex-wrap: wrap;
}}

.hud-label  {{ color: var(--text-muted); font-weight: 600; letter-spacing: 0.06em; }}
.hud-value  {{ color: var(--text-secondary); }}
.hud-active {{ color: var(--success); }}
.hud-none   {{ color: var(--danger); }}

/* ── DEPLOYMENT REGIME CARDS ── */
.regime-cloud  {{ border-color: #c7d2fe; }}
.regime-edge   {{ border-color: {COLORS['RedL']}; }}
.regime-mobile {{ border-color: {COLORS['OrangeL']}; }}
.regime-tiny   {{ border-color: {COLORS['GreenL']}; }}

/* ── VISCERAL ANIMATIONS ── */
@keyframes shake-hard {{
    0% {{ transform: translate(1px, 1px) rotate(0deg); }}
    10% {{ transform: translate(-1px, -2px) rotate(-1deg); }}
    20% {{ transform: translate(-3px, 0px) rotate(1deg); }}
    30% {{ transform: translate(3px, 2px) rotate(0deg); }}
    40% {{ transform: translate(1px, -1px) rotate(1deg); }}
    50% {{ transform: translate(-1px, 2px) rotate(-1deg); }}
    60% {{ transform: translate(-3px, 1px) rotate(0deg); }}
    70% {{ transform: translate(3px, 1px) rotate(-1deg); }}
    80% {{ transform: translate(-1px, -1px) rotate(1deg); }}
    90% {{ transform: translate(1px, 2px) rotate(0deg); }}
    100% {{ transform: translate(1px, -2px) rotate(-1deg); }}
}}
.shake-hard {{
    animation: shake-hard 0.4s cubic-bezier(.36,.07,.19,.97) both;
}}

@keyframes pulse-danger {{
    0% {{ box-shadow: 0 0 0 0 rgba(203, 32, 45, 0.7); }}
    70% {{ box-shadow: 0 0 0 10px rgba(203, 32, 45, 0); }}
    100% {{ box-shadow: 0 0 0 0 rgba(203, 32, 45, 0); }}
}}
.pulse-danger {{
    animation: pulse-danger 1.5s infinite;
    border: 2px solid var(--danger) !important;
}}

/* ── ORIENTATION COMPLETE BANNER ── */
.orientation-complete {{
    background: var(--surface-0);
    border-radius: var(--radius-md);
    padding: 16px 22px;
    margin-top: 16px;
    border: 1px solid var(--surface-1);
    display: flex;
    align-items: center;
    gap: 16px;
    font-size: 0.87rem;
    color: var(--text-muted);
    line-height: 1.6;
}}
</style>
""")

# ── HELPER FUNCTIONS ──────────────────────────────────────────────────────────

def progress_bar(current: int, total: int, labels: list[str] = None) -> mo.Html:
    """
    Renders a step-progress indicator for use at the start of each concept section.

    Args:
        current: 0-indexed step currently active (0 = first step)
        total:   total number of steps
        labels:  optional list of step labels (len must equal total)

    Example:
        progress_bar(1, 3, ["95% Problem", "Constraints", "Regimes"])
    """
    import marimo as mo
    items = []
    for i in range(total):
        if i < current:
            dot_class = "lab-progress-dot done"
            line_class = "lab-progress-line done"
        elif i == current:
            dot_class = "lab-progress-dot active"
            line_class = "lab-progress-line"
        else:
            dot_class = "lab-progress-dot"
            line_class = "lab-progress-line"

        label_html = ""
        if labels and i < len(labels):
            color = "#008F45" if i < current else ("#006395" if i == current else "#94a3b8")
            label_html = f'<span style="color:{color}; font-size:0.72rem; font-weight:700; white-space:nowrap;">{labels[i]}</span>'

        items.append(f'<span class="{dot_class}"></span>')
        if label_html:
            items.append(label_html)
        if i < total - 1:
            items.append(f'<span class="{line_class}"></span>')

    inner = "".join(items)
    return mo.Html(f'<div class="lab-progress">{inner}</div>')


def concept_section_header(step: int, total: int, title: str, subtitle: str = "") -> mo.Html:
    """
    Renders a section header with step counter, title, and optional subtitle.
    Use at the top of each concept block.

    Example:
        concept_section_header(1, 3, "The 95% Problem", "ML Systems ≠ ML Models")
    """
    import marimo as mo
    sub_html = f'<div style="color:#94a3b8; font-size:0.82rem; margin-top:3px;">{subtitle}</div>' if subtitle else ""
    return mo.Html(f"""
    <div style="margin: 1.5rem 0 0.5rem 0;">
        <div class="concept-header">
            <span style="background:#006395; color:white; border-radius:50%;
                         width:20px; height:20px; display:inline-flex; align-items:center;
                         justify-content:center; font-size:0.72rem; font-weight:800;
                         flex-shrink:0;">{step}</span>
            <span>Step {step} of {total}</span>
            <span class="concept-header-line"></span>
        </div>
        <div style="font-size:1.35rem; font-weight:800; color:#0f172a; line-height:1.2;">{title}</div>
        {sub_html}
    </div>
    """)


def confidence_widget(mo_instance) -> "mo.ui.radio":
    """
    Returns a compact confidence-rating radio to place beneath each check question.
    The value is stored in Design Ledger as metadata alongside check answers.

    Usage:
        conf = confidence_widget(mo)
        # render next to check:  mo.hstack([check_radio, conf])
    """
    return mo_instance.ui.radio(
        options={
            "🤔  Not sure":        "low",
            "🙂  Pretty confident": "med",
            "😤  Very sure":       "high",
        },
        label="Confidence:",
        inline=True,
    )


def apply_plotly_theme(fig):
    """Apply standard MLSys plot aesthetics."""
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font_family="Inter, sans-serif",
        font_color=COLORS["Text"],
        margin=dict(l=40, r=20, t=20, b=40),
        xaxis=dict(gridcolor="#f1f5f9", linecolor=COLORS["Border"]),
        yaxis=dict(gridcolor="#f1f5f9", linecolor=COLORS["Border"]),
    )
    return fig

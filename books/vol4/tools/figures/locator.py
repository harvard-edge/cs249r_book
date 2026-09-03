import os
import subprocess

NAVY = "#1F407A"
BLUE = "#215CAF"
PETROL = "#007A87"
TEAL = "#10B981"
BRONZE = "#B87333"
AMBER = "#D97706"
CRIMSON = "#A51C30"
CORAL = "#DC2626"
PURPLE = "#5B4B8A"
SLATE = "#475569"
MUTED = "#64748B"
INK = "#0F172A"
BG_LIGHT = "#F8FAFC"
BG_WHITE = "#FFFFFF"
BORDER = "#CBD5E1"
BORDER_DARK = "#94A3B8"

COMMON_STYLE = """
<style>
  text { font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; }
  .section-hdr { font-size: 11px; font-weight: 700; fill: #1F407A; letter-spacing: 0.05em; }
  .badge-text { font-size: 9px; font-weight: 700; text-anchor: middle; letter-spacing: 0.03em; }
</style>
"""

COMMON_DEFS = """
<defs>
  <marker id="arr-navy" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
    <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#1F407A"/>
  </marker>
  <marker id="arr-slate" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
    <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#94A3B8"/>
  </marker>
  <marker id="arr-crimson" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
    <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#A51C30"/>
  </marker>
</defs>
"""

def gen_pipeline_locator(active_stage, active_pill, target_svg_path):
    W = 880
    H = 115
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="100%" height="100%">']
    svg.append(COMMON_STYLE)
    svg.append(COMMON_DEFS)
    svg.append(f'<rect width="{W}" height="{H}" fill="{BG_WHITE}" stroke="{BORDER}" stroke-width="1"/>')

    # Top Header
    svg.append(f'<text x="24" y="24" class="section-hdr">END-TO-END EMBODIED PIPELINE LOCATOR</text>')

    # Active Pill Badge on top right (clean 90-degree rectangle)
    pill_w = len(active_pill) * 6.5 + 20
    pill_x = W - 24 - pill_w
    svg.append(f'<rect x="{pill_x}" y="12" width="{pill_w}" height="18" fill="{NAVY}" fill-opacity="0.10" stroke="{NAVY}" stroke-width="0.8"/>')
    svg.append(f'<text x="{pill_x+pill_w/2}" y="24" class="badge-text" fill="{NAVY}">{active_pill}</text>')

    # 7 Pipeline Stage Cards
    stages = [
        ("TRANSDUCTION", "Photons → Bits"),
        ("PERCEPTION", "Spatial Tokens"),
        ("MEMORY", "SE(3) World Model"),
        ("REASONING", "Intent Leases"),
        ("PLANNING", "Action Chunks"),
        ("REFLEXES", "Safety Enforcer"),
        ("PHYSICS", "Plant Dynamics")
    ]

    card_w = 106
    gap = 8
    start_x = (W - (7 * card_w + 6 * gap)) / 2
    card_y = 38
    card_h = 62

    for i, (name, sub) in enumerate(stages):
        cx = start_x + i * (card_w + gap)
        is_active = (i == active_stage)

        if is_active:
            stroke_col = NAVY
            stroke_w = "1.8"
            bg_col = f"{NAVY}12"
            txt_col = NAVY
            weight = "700"
            sub_col = INK
        else:
            stroke_col = BORDER
            stroke_w = "0.9"
            bg_col = BG_LIGHT
            txt_col = MUTED
            weight = "600"
            sub_col = MUTED

        # Sharp 90-degree card rectangle
        svg.append(f'<rect x="{cx}" y="{card_y}" width="{card_w}" height="{card_h}" fill="{bg_col}" stroke="{stroke_col}" stroke-width="{stroke_w}"/>')

        if is_active:
            # Top highlight accent bar (sharp 90-degree)
            svg.append(f'<rect x="{cx}" y="{card_y}" width="{card_w}" height="3.5" fill="{NAVY}"/>')

        svg.append(f'<text x="{cx+card_w/2}" y="{card_y+25}" font-size="8.5" font-weight="{weight}" fill="{txt_col}" text-anchor="middle">{name}</text>')
        svg.append(f'<text x="{cx+card_w/2}" y="{card_y+45}" font-size="8" fill="{sub_col}" text-anchor="middle">{sub}</text>')

        if i < 6:
            ax1 = cx + card_w + 1
            ax2 = ax1 + gap - 2
            ay = card_y + card_h / 2
            arr_col = NAVY if (is_active or i == active_stage - 1) else BORDER_DARK
            marker = "url(#arr-navy)" if arr_col == NAVY else "url(#arr-slate)"
            svg.append(f'<line x1="{ax1}" y1="{ay}" x2="{ax2}" y2="{ay}" stroke="{arr_col}" stroke-width="1.2" marker-end="{marker}"/>')

    svg.append('</svg>')
    
    from .common import save_svg_and_pdf
    save_svg_and_pdf(target_svg_path, "\n".join(svg))
    print(f"Generated clean 90-deg locator: {target_svg_path}")

def run_all():
    locators = [
        (0, "CHAPTER 01 · THE BOUNDARY", "book/chapters/01-boundary/figures/fig_pipeline_locator.svg"),
        (0, "CHAPTER 02 · THE FIVE PHYSICAL CONSTRAINTS", "book/chapters/02-body/figures/fig_pipeline_locator.svg"),
        (1, "CHAPTER 03 · FOUNDATIONS OF COGNITIVE AGENCY", "book/chapters/03-brain/figures/fig_pipeline_locator.svg"),
        (2, "CHAPTER 04 · MULTI-RATE SYSTEM HIERARCHY", "book/chapters/04-nervous/figures/fig_pipeline_locator.svg"),
        (1, "CHAPTER 05 · SPATIAL PERCEPTION & TRANSDUCTION", "book/chapters/05-data/figures/fig_pipeline_locator.svg"),
        (2, "CHAPTER 06 · TEMPORAL MEMORY & WORLD MODELS", "book/chapters/06-training/figures/fig_pipeline_locator.svg"),
        (3, "CHAPTER 07 · INTENT & SEMANTIC REASONING", "book/chapters/07-evaluation/figures/fig_pipeline_locator.svg"),
        (4, "CHAPTER 08 · ACTION GENERATION & TRAJECTORY PLANNING", "book/chapters/08-perception/figures/fig_pipeline_locator.svg"),
        (5, "CHAPTER 09 · REAL-TIME SAFETY ENFORCEMENT & REFLEXES", "book/chapters/09-memory/figures/fig_pipeline_locator.svg"),
        (4, "CHAPTER 10 · HETEROGENEOUS COMPUTE PLACEMENT", "book/chapters/10-intent/figures/fig_pipeline_locator.svg"),
        (5, "CHAPTER 11 · RUNTIME GOVERNANCE & HUMAN AUTHORITY", "book/chapters/11-planning/figures/fig_pipeline_locator.svg"),
        (5, "CHAPTER 12 · WHOLE-SYSTEM QUALIFICATION & ASSURANCE", "book/chapters/12-enforcement/figures/fig_pipeline_locator.svg"),
        (6, "CHAPTER 13 · FRONTIER COGNITION & CAPSTONE INTEGRATION", "book/chapters/13-placement/figures/fig_pipeline_locator.svg")
    ]

    for stage, pill, path in locators:
        gen_pipeline_locator(stage, pill, path)

if __name__ == "__main__":
    run_all()

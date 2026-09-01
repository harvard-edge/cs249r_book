"""
book/tools/figures/common.py
Shared design system, colors, fonts, and SVG utilities.
Harvard Crimson & ETH Zurich Academic Semantic Palette.
"""

import os
import subprocess

# Harvard Crimson & ETH Zurich Academic Semantic Palette
NAVY = "#1F407A"         # Structural Primary / MPU / System 2
BLUE = "#215CAF"         # MPU / Ingestion / Deliberation
PETROL = "#007A87"       # MCU / Real-Time Reflex / System 1
TEAL = "#10B981"         # Safe / Release / Deploy / Verified
BRONZE = "#B87333"       # Trajectory Planning / System 1.5 / Action Chunking
AMBER = "#D97706"        # Warning / Shared Control / Condition
CRIMSON = "#A51C30"      # Physical World / Hazard / Kinetic Energy / Refuse
CORAL = "#DC2626"        # Fault / E-Stop / Invariant Violation
PURPLE = "#5B4B8A"       # Governance / Authority / Safety Case
SLATE = "#475569"        # Body text / Secondary lines
MUTED = "#64748B"        # Subtitles / Secondary labels
INK = "#1A202C"          # Dark Title text
BG_LIGHT = "#F8FAFC"     # Card Background
BG_WHITE = "#FFFFFF"     # Container Background
BORDER = "#CBD5E1"       # Subtle card border
BORDER_DARK = "#94A3B8"  # Prominent border

COMMON_STYLE = """
    <style>
      text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; }
      .title { font-weight: 700; font-size: 15px; fill: #1F407A; text-anchor: middle; letter-spacing: 0.5px; }
      .subtitle { font-size: 11px; fill: #64748B; text-anchor: middle; }
      .section-hdr { font-weight: 700; font-size: 13px; fill: #1F407A; letter-spacing: 0.3px; }
      .card-title { font-weight: 700; font-size: 11.5px; }
      .body-text { font-size: 10px; fill: #475569; }
      .bold-text { font-weight: 600; font-size: 10px; fill: #1A202C; }
      .small-text { font-size: 9px; fill: #64748B; }
      .code-text { font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, Courier, monospace; font-size: 9px; fill: #1F407A; font-weight: 600; }
      .badge-text { font-weight: 700; font-size: 9px; text-anchor: middle; }
    </style>
"""

COMMON_DEFS = """
    <defs>
      <marker id="arr-blue" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
        <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#215CAF"/>
      </marker>
      <marker id="arr-navy" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
        <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#1F407A"/>
      </marker>
      <marker id="arr-petrol" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
        <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#007A87"/>
      </marker>
      <marker id="arr-teal" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
        <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#10B981"/>
      </marker>
      <marker id="arr-bronze" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
        <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#B87333"/>
      </marker>
      <marker id="arr-crimson" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
        <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#A51C30"/>
      </marker>
      <marker id="arr-coral" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
        <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#DC2626"/>
      </marker>
      <marker id="arr-purple" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
        <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#5B4B8A"/>
      </marker>
      <marker id="arr-slate" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
        <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#64748B"/>
      </marker>
      <filter id="shadow" x="-2%" y="-3%" width="104%" height="108%" filterUnits="userSpaceOnUse">
        <feDropShadow dx="0" dy="2" stdDeviation="3" flood-color="#0F172A" flood-opacity="0.05"/>
      </filter>
    </defs>
"""

import re
import xml.etree.ElementTree as ET

def sanitize_svg_xml(svg_str):
    def fix_text_body(match):
        open_tag = match.group(1)
        body = match.group(2)
        close_tag = match.group(3)
        body = re.sub(r'&(?!(?:amp|lt|gt|quot|apos|#\d+|#x[0-9a-fA-F]+);)', '&amp;', body)
        body = re.sub(r'<(?!\/?tspan\b)', '&lt;', body)
        return f'{open_tag}{body}{close_tag}'

    pattern = re.compile(r'(<text\b[^>]*>)(.*?)(</text>)', re.DOTALL)
    return pattern.sub(fix_text_body, svg_str)

def save_svg_and_pdf(path, content):
    # Ensure path correctly resolves to book/chapters
    if not os.path.isabs(path):
        cwd = os.getcwd()
        if cwd.endswith("/book") and path.startswith("book/"):
            path = path[5:]
        elif not cwd.endswith("/book") and not path.startswith("book/"):
            path = os.path.join("book", path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    content = sanitize_svg_xml(content)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content.strip() + "\n")
    pdf_path = os.path.splitext(path)[0] + ".pdf"
    res = subprocess.run(["rsvg-convert", "-f", "pdf", "-o", pdf_path, path], capture_output=True)
    if res.returncode != 0:
        subprocess.run(["inkscape", "--export-filename=" + pdf_path, path], capture_output=True)
    print(f"Generated: {path} and {pdf_path}")



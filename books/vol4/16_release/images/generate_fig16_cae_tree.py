import os

def create_svg():
    width = 900
    height = 680
    
    # Colors
    c_blue = "#1F407A"
    c_light_grey = "#E2E8F0"
    c_dark_grey = "#2D3748"
    c_red = "#A51C30"
    c_white = "#FFFFFF"
    
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="100%">
    <defs>
        <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="{c_dark_grey}" />
        </marker>
        <marker id="arrow-red" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="{c_red}" />
        </marker>
        <marker id="arrow-blue" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="{c_blue}" />
        </marker>
    </defs>
    
    <!-- Background -->
    <rect width="100%" height="100%" fill="{c_white}" />
    
    <style>
        .claim-title {{ font-family: sans-serif; font-weight: bold; font-size: 14px; fill: {c_white}; }}
        .claim-text {{ font-family: sans-serif; font-size: 13px; fill: {c_dark_grey}; }}
        .box-title {{ font-family: sans-serif; font-weight: bold; font-size: 14px; fill: {c_dark_grey}; }}
        .box-text {{ font-family: sans-serif; font-size: 13px; fill: {c_dark_grey}; }}
        .white-text {{ font-family: sans-serif; font-size: 13px; fill: {c_white}; }}
        .bold-text {{ font-family: sans-serif; font-weight: bold; font-size: 13px; fill: {c_dark_grey}; }}
        .red-text {{ font-family: sans-serif; font-weight: bold; font-size: 13px; fill: {c_red}; }}
        .blue-text {{ font-family: sans-serif; font-weight: bold; font-size: 13px; fill: {c_blue}; }}
    </style>
"""

    def box(x, y, w, h, title, lines, fill=c_light_grey, stroke=c_dark_grey, title_fill=c_blue, text_class="box-text", title_class="claim-title", title_bg=c_blue, stroke_width=2):
        res = f"""
    <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="4" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}"/>
    <rect x="{x}" y="{y}" width="{w}" height="24" rx="4" fill="{title_bg}" />
    <path d="M {x} {y+20} L {x+w} {y+20} L {x+w} {y+24} L {x} {y+24} Z" fill="{title_bg}" />
    <text x="{x+10}" y="{y+16}" class="{title_class}">{title}</text>
"""
        ly = y + 42
        for line in lines:
            res += f'    <text x="{x+10}" y="{ly}" class="{text_class}">{line}</text>\n'
            ly += 18
        return res

    def path(x1, y1, x2, y2, color=c_dark_grey, marker="arrow", dash=""):
        d_attr = ""
        if dash:
            d_attr = f'stroke-dasharray="{dash}"'
        return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="2" marker-end="url(#{marker})" {d_attr}/>'

    # Lines
    # C1 to C1.1, C1.2, C1.3
    svg += path(450, 120, 160, 170) # to C1.1
    svg += path(450, 120, 450, 170) # to C1.2
    svg += path(450, 120, 740, 170) # to C1.3
    
    # D1 to C1.1
    svg += path(160, 310, 160, 260, color=c_red, marker="arrow-red")
    
    # D1 to Fallback
    svg += path(160, 400, 310, 400, color=c_dark_grey, dash="5,5")
    
    # C1.2 to Fallback
    svg += path(450, 250, 450, 350)
    
    # Fallback to Adjudication
    svg += path(450, 450, 450, 500, color=c_blue, marker="arrow-blue")

    # Boxes
    svg += box(250, 40, 400, 80, "Claim C1: Operational Safety", 
               ["Normal force ≤ 50 N, Shear force ≤ 20 N", "Maintained across all operational states and faults"])
               
    svg += box(50, 170, 220, 90, "Sub-Claim C1.1: Tactile Reflex",
               ["Soft tactile slip detector", "halts arm before normal", "force reaches 14 N"])
               
    svg += box(340, 170, 220, 90, "Sub-Claim C1.2: Torque Tripwire",
               ["Secondary torque tripwire", "halts arm at F_trip = 35 N"])
               
    svg += box(630, 170, 220, 90, "Sub-Claim C1.3: Deterministic Exec",
               ["Nervous system hardware", "guarantees loop latency", "≤ 25 ms (worst-case)"])
               
    svg += box(50, 310, 220, 110, "Defeater D1: Adversarial Audit",
               ["Residual cutting oil drops", "friction (μ = 0.85 → 0.12).", "Workpiece slips silently", "without elastomer strain.", "Reflex fails!"],
               stroke=c_red, title_bg=c_red)
               
    svg += box(310, 350, 280, 100, "Unconditional Release REFUSED",
               ["Primary reflex disabled.", "Unmitigated secondary tripwire", "allows peak force = 88 N.", "88 N > 50 N (Claim Violation)"],
               stroke=c_red, title_bg=c_dark_grey)

    svg += box(250, 500, 400, 130, "Adjudication: Operate with Conditions",
               ["Binding Runtime Contracts:",
                "1. Restrict payload mass: m ≤ 1.5 kg",
                "2. Automated optical surface-dryness check",
                "3. Tighten torque tripwire: F_trip = 18 N",
                "Result: Peak impact F_peak = 38 N ≤ 50 N"],
               fill="#F0F4F8", stroke=c_blue, title_bg=c_blue)
               
    # Add text labels on arrows
    svg += f'<text x="110" y="295" class="red-text" transform="rotate(-90 150 295)">DEFEATS</text>\n'
    svg += f'<text x="210" y="390" class="bold-text">Fallback evaluates</text>\n'

    svg += "</svg>"
    
    with open("books/vol4/16_release/images/svg/fig16_cae_tree.svg", "w") as f:
        f.write(svg)

if __name__ == "__main__":
    create_svg()

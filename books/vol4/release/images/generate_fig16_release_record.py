import os

def create_svg():
    width = 980
    height = 500
    
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
        <marker id="arrow-blue" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="{c_blue}" />
        </marker>
        <marker id="arrow-red" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="{c_red}" />
        </marker>
    </defs>
    
    <!-- Background -->
    <rect width="100%" height="100%" fill="{c_white}" />
    
    <style>
        .box-title {{ font-family: sans-serif; font-weight: bold; font-size: 15px; fill: {c_white}; }}
        .sub-title {{ font-family: sans-serif; font-weight: bold; font-size: 13px; fill: {c_dark_grey}; }}
        .box-text {{ font-family: sans-serif; font-size: 13px; fill: {c_dark_grey}; }}
        .mono-text {{ font-family: monospace; font-size: 12px; fill: {c_dark_grey}; }}
        .red-text {{ font-family: sans-serif; font-weight: bold; font-size: 13px; fill: {c_red}; }}
        .blue-text {{ font-family: sans-serif; font-weight: bold; font-size: 13px; fill: {c_blue}; }}
    </style>
"""

    def box(x, y, w, h, title, fill=c_light_grey, stroke=c_dark_grey, title_bg=c_blue, stroke_width=2):
        return f"""
    <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="4" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}"/>
    <rect x="{x}" y="{y}" width="{w}" height="28" rx="4" fill="{title_bg}" />
    <path d="M {x} {y+24} L {x+w} {y+24} L {x+w} {y+28} L {x} {y+28} Z" fill="{title_bg}" />
    <text x="{x+10}" y="{y+19}" class="box-title">{title}</text>
"""
    
    def path(x1, y1, x2, y2, color=c_dark_grey, marker="arrow", dash="", thick=2):
        d_attr = ""
        if dash:
            d_attr = f'stroke-dasharray="{dash}"'
        return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{thick}" marker-end="url(#{marker})" {d_attr}/>'

    # Arrows between main blocks
    svg += path(300, 220, 360, 220, color=c_blue, marker="arrow-blue", thick=3)
    svg += f'<text x="315" y="210" class="blue-text">Verifies</text>\n'
    
    svg += path(620, 220, 680, 220, color=c_blue, marker="arrow-blue", thick=3)
    svg += f'<text x="625" y="210" class="blue-text">Permits</text>\n'

    # Block 1: Release Record
    svg += box(40, 80, 260, 300, "1. Cryptographic Release Record", fill="#F8FAFC", stroke=c_blue)
    svg += f'<text x="55" y="130" class="sub-title">Immutable Engineering Manifest</text>\n'
    
    y = 155
    items = [
        ("Claim ID:", "req-tactile-09"),
        ("GSN Topology Hash:", "0x8F4A...D21C"),
        ("Evidence Pointers:", "verified_trace_12"),
        ("Policy Weights:", "0x3B9C...11EF"),
        ("FPGA Bitstream:", "0xA11F...402B"),
        ("RTOS Kernel:", "0x77C1...99AA")
    ]
    for k, v in items:
        svg += f'<text x="55" y="{y}" class="box-text">{k}</text>\n'
        svg += f'<text x="175" y="{y}" class="mono-text">{v}</text>\n'
        y += 24
        
    svg += f'<rect x="55" y="320" width="230" height="40" rx="2" fill="{c_light_grey}" stroke="{c_dark_grey}" />\n'
    svg += f'<text x="65" y="344" class="box-text" font-weight="bold">Adjudicator Signature: VALID</text>\n'

    # Block 2: Boot Verification
    svg += box(360, 80, 260, 300, "2. Hardware Root of Trust", fill="#F8FAFC", stroke=c_blue)
    
    svg += f'<rect x="380" y="125" width="220" height="40" rx="2" fill="{c_white}" stroke="{c_dark_grey}" />\n'
    svg += f'<text x="395" y="150" class="box-text" font-weight="bold">Silicon OTP eFuses (Keys)</text>\n'
    
    svg += path(490, 165, 490, 195)
    
    svg += f'<rect x="380" y="195" width="220" height="60" rx="2" fill="{c_light_grey}" stroke="{c_dark_grey}" />\n'
    svg += f'<text x="395" y="215" class="sub-title">Bootloader Verification</text>\n'
    svg += f'<text x="395" y="235" class="box-text">Validates hashes & signatures</text>\n'
    svg += f'<text x="395" y="250" class="box-text">against eFuse public keys</text>\n'
    
    svg += path(490, 255, 490, 285)
    
    svg += f'<rect x="380" y="285" width="220" height="45" rx="2" fill="#E6FFFA" stroke="#319795" />\n'
    svg += f'<text x="395" y="305" class="box-text" font-weight="bold" fill="#285E61">Energize Motor Inverters</text>\n'
    svg += f'<text x="395" y="320" class="box-text" fill="#285E61">(Hardware Permit Asserted)</text>\n'

    # Block 3: Runtime Invalidation
    svg += box(680, 80, 260, 300, "3. Runtime Nervous System", fill="#FFF5F5", stroke=c_red, title_bg=c_red)
    
    svg += f'<text x="695" y="130" class="sub-title">Continuous Envelope Monitor</text>\n'
    svg += f'<text x="695" y="150" class="box-text">1000 Hz Hard Real-Time Loop</text>\n'
    
    svg += f'<rect x="695" y="165" width="230" height="75" rx="2" fill="{c_white}" stroke="{c_dark_grey}" />\n'
    svg += f'<text x="705" y="185" class="box-text" font-weight="bold">Envelope Trichotomy Eval</text>\n'
    svg += f'<text x="705" y="205" class="box-text">✓ Known-True (Permitted)</text>\n'
    svg += f'<text x="705" y="220" class="box-text" fill="{c_red}">✗ Known-False / Unknown</text>\n'
    
    svg += path(810, 240, 810, 270, color=c_red, marker="arrow-red", thick=2)
    svg += f'<text x="820" y="260" class="red-text">Violation / Epistemic Gap</text>\n'
    
    svg += f'<rect x="695" y="270" width="230" height="70" rx="2" fill="{c_red}" stroke="{c_dark_grey}" />\n'
    svg += f'<text x="705" y="295" class="box-title">LATCH INTERLOCKS</text>\n'
    svg += f'<text x="705" y="315" class="box-title" font-size="13px">Revoke Operating Authority</text>\n'

    svg += "</svg>"
    
    with open("books/vol4/release/images/svg/fig16_release_record.svg", "w") as f:
        f.write(svg)

if __name__ == "__main__":
    create_svg()

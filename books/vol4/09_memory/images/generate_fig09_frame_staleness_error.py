import os
import math

def generate_svg():
    width = 800
    height = 500
    
    # Colors
    c_dark_blue = "#1F407A"
    c_grey_light = "#E2E8F0"
    c_grey_dark = "#2D3748"
    c_red = "#A51C30"
    c_blue_light = "#3182CE"
    c_bg = "#FFFFFF"

    # SVG header
    svg = [
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg" style="background-color: {c_bg}; font-family: sans-serif;">',
        '  <defs>',
        '    <marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">',
        f'      <path d="M0,0 L0,6 L9,3 z" fill="{c_grey_dark}" />',
        '    </marker>',
        '    <marker id="arrow_red" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">',
        f'      <path d="M0,0 L0,6 L9,3 z" fill="{c_red}" />',
        '    </marker>',
        '    <marker id="arrow_blue" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">',
        f'      <path d="M0,0 L0,6 L9,3 z" fill="{c_dark_blue}" />',
        '    </marker>',
        '  </defs>',
    ]
    
    # Base layout
    base_y = 400
    base_x_start = 150
    base_x_end = 350
    
    # Time t position (Base 1)
    x1 = 200
    y1 = base_y
    l = 300 # length of arm visual
    angle1 = -math.pi / 2 # straight up
    
    tx1 = x1 + l * math.cos(angle1)
    ty1 = y1 + l * math.sin(angle1)
    
    # Time t + dt position (Base 2)
    # Translation = 100 px (representing 3.2 cm)
    x2 = x1 + 100
    y2 = y1
    angle2 = angle1 + 0.3 # rotation (representing 3.6 cm arc)
    
    tx2 = x2 + l * math.cos(angle2)
    ty2 = y2 + l * math.sin(angle2)
    
    # Draw arm at t (Ghosted / Previous)
    svg.append(f'  <!-- Time t -->')
    svg.append(f'  <line x1="{x1}" y1="{y1}" x2="{tx1}" y2="{ty1}" stroke="{c_grey_light}" stroke-width="8" stroke-dasharray="10,5"/>')
    svg.append(f'  <circle cx="{x1}" cy="{y1}" r="12" fill="{c_grey_light}"/>')
    svg.append(f'  <circle cx="{tx1}" cy="{ty1}" r="8" fill="{c_grey_light}"/>')
    
    # Draw translation of base
    svg.append(f'  <line x1="{x1}" y1="{y1+30}" x2="{x2-5}" y2="{y1+30}" stroke="{c_dark_blue}" stroke-width="2" marker-end="url(#arrow_blue)"/>')
    svg.append(f'  <text x="{(x1+x2)/2}" y="{y1+50}" font-size="14" fill="{c_dark_blue}" text-anchor="middle">e_trans = 3.20 cm</text>')
    
    # Intermediate arm position (translation only, no rotation)
    tx_int = x2 + l * math.cos(angle1)
    ty_int = y2 + l * math.sin(angle1)
    svg.append(f'  <!-- Translation only -->')
    svg.append(f'  <line x1="{x2}" y1="{y2}" x2="{tx_int}" y2="{ty_int}" stroke="{c_grey_light}" stroke-width="8" stroke-dasharray="4,4"/>')
    svg.append(f'  <circle cx="{tx_int}" cy="{ty_int}" r="8" fill="{c_grey_light}"/>')
    
    # Draw rotation vector from intermediate tool point to final tool point
    # Arc path
    r_arc = l
    # svg.append(f'  <path d="M {tx_int} {ty_int} A {r_arc} {r_arc} 0 0 1 {tx2} {ty2}" fill="none" stroke="{c_dark_blue}" stroke-width="2" marker-end="url(#arrow_blue)"/>')
    
    # Let's draw arc for base rotation
    arc_radius = 80
    ax1 = x2 + arc_radius * math.cos(angle1)
    ay1 = y2 + arc_radius * math.sin(angle1)
    ax2 = x2 + arc_radius * math.cos(angle2)
    ay2 = y2 + arc_radius * math.sin(angle2)
    svg.append(f'  <path d="M {ax1} {ay1} A {arc_radius} {arc_radius} 0 0 1 {ax2} {ay2}" fill="none" stroke="{c_dark_blue}" stroke-width="2" marker-end="url(#arrow_blue)"/>')
    svg.append(f'  <text x="{x2 + 30}" y="{y2 - 90}" font-size="14" fill="{c_dark_blue}" text-anchor="start">Δθ (1.50 rad/s)</text>')
    
    # Draw arm at t + dt (Final position)
    svg.append(f'  <!-- Time t+dt -->')
    svg.append(f'  <line x1="{x2}" y1="{y2}" x2="{tx2}" y2="{ty2}" stroke="{c_dark_blue}" stroke-width="8"/>')
    svg.append(f'  <circle cx="{x2}" cy="{y2}" r="12" fill="{c_dark_blue}"/>')
    svg.append(f'  <circle cx="{tx2}" cy="{ty2}" r="8" fill="{c_dark_blue}"/>')
    
    # Tool point error vectors
    # Translation error at top
    svg.append(f'  <line x1="{tx1}" y1="{ty1-20}" x2="{tx_int-5}" y2="{ty_int-20}" stroke="{c_grey_dark}" stroke-width="2" marker-end="url(#arrow)"/>')
    svg.append(f'  <text x="{(tx1+tx_int)/2}" y="{ty1-30}" font-size="14" fill="{c_grey_dark}" text-anchor="middle">Translation</text>')
    
    # Rotation error at top
    svg.append(f'  <path d="M {tx_int} {ty_int-20} A {r_arc+20} {r_arc+20} 0 0 1 {tx2} {ty2-20}" fill="none" stroke="{c_grey_dark}" stroke-width="2" marker-end="url(#arrow)"/>')
    svg.append(f'  <text x="{tx2+30}" y="{ty2-30}" font-size="14" fill="{c_grey_dark}" text-anchor="start">e_rot ≈ ℓ Δθ = 3.60 cm</text>')
    
    # Total error arrow
    svg.append(f'  <line x1="{tx1}" y1="{ty1}" x2="{tx2-5}" y2="{ty2-5}" stroke="{c_red}" stroke-width="2" marker-end="url(#arrow_red)"/>')
    svg.append(f'  <text x="{(tx1+tx2)/2}" y="{ty1 + 35}" font-size="16" fill="{c_red}" font-weight="bold" text-anchor="middle">Total Error: e_tool = 6.80 cm</text>')
    
    # Add arm length label
    mid_x = x2 + (l/2) * math.cos(angle2) + 20
    mid_y = y2 + (l/2) * math.sin(angle2)
    svg.append(f'  <text x="{mid_x}" y="{mid_y}" font-size="14" fill="{c_dark_blue}" text-anchor="start">ℓ = 0.60 m</text>')
    
    # Labels
    svg.append(f'  <text x="{x1}" y="{y1+70}" font-size="14" fill="{c_grey_dark}" font-weight="bold" text-anchor="middle">t = 0 (Reported State)</text>')
    svg.append(f'  <text x="{x2}" y="{y1+95}" font-size="14" fill="{c_dark_blue}" font-weight="bold" text-anchor="middle">t = 40 ms (True State)</text>')
    
    # Workpiece clearance line
    wx1 = tx1 - 30
    wx2 = tx1 + 30
    wy = ty1 - 10
    svg.append(f'  <line x1="{wx1}" y1="{wy}" x2="{wx2}" y2="{wy}" stroke="{c_grey_dark}" stroke-width="4"/>')
    svg.append(f'  <rect x="{wx1}" y="{wy-15}" width="60" height="15" fill="{c_grey_light}"/>')
    svg.append(f'  <text x="{wx1}" y="{wy-20}" font-size="12" fill="{c_grey_dark}">Workpiece Target</text>')
    
    # Collision zone
    svg.append(f'  <circle cx="{tx2}" cy="{ty2}" r="25" fill="{c_red}" fill-opacity="0.2" stroke="{c_red}" stroke-dasharray="4,4"/>')
    svg.append(f'  <text x="{tx2+35}" y="{ty2+10}" font-size="14" fill="{c_red}" font-weight="bold">Collision</text>')
    
    # Title/Description text box
    svg.append(f'  <rect x="20" y="20" width="300" height="80" rx="4" ry="4" fill="{c_grey_light}"/>')
    svg.append(f'  <text x="30" y="45" font-size="16" fill="{c_dark_blue}" font-weight="bold">Coordinate Staleness Error</text>')
    svg.append(f'  <text x="30" y="65" font-size="14" fill="{c_grey_dark}">v = 0.80 m/s</text>')
    svg.append(f'  <text x="30" y="85" font-size="14" fill="{c_grey_dark}">ω = 1.50 rad/s, Δt = 40 ms</text>')
    
    svg.append('</svg>')
    
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'svg')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'fig09_frame_staleness_error.svg')
    with open(out_path, 'w') as f:
        f.write('\\n'.join(svg))
    print("Generated:", out_path)

if __name__ == "__main__":
    generate_svg()

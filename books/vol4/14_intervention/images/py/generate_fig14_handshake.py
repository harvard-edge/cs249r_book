import svgwrite

def draw_fsm_fig():
    width, height = 900, 600
    dwg = svgwrite.Drawing('books/vol4/14_intervention/images/svg/fig14_authority_handshake_fsm.svg', size=(width, height))

    # Colors
    bg = "#FFFFFF"
    dark_blue = "#1F407A"
    grey_bg = "#F7FAFC"
    stroke_grey = "#CBD5E1"
    dark_grey = "#2D3748"
    deep_red = "#A51C30"
    
    dwg.add(dwg.rect(insert=(0,0), size=(width, height), fill=bg))
    
    # Fonts
    font_family = "system-ui, -apple-system, sans-serif"
    
    # -----------------------------
    # Top: 4-phase Pipeline
    # -----------------------------
    pipeline_y = 50
    box_w, box_h = 200, 100
    spacing = 20
    start_x = (width - (4*box_w + 3*spacing)) / 2
    
    phases = [
        {"title": "1. Request", "desc": "Override torque |τ| > 4.0 Nm\nor policy ODD departure"},
        {"title": "2. Intercept & Auth", "desc": "1 kHz hardware token\nFreshness Δt ≤ 100ms"},
        {"title": "3. Commit & Blend", "desc": "Atomic register swap\nC² quintic blending α(t)"},
        {"title": "4. Confirm & Active", "desc": "α = 1.0 active human cmd\nLease T_lease ≤ 100ms"}
    ]
    
    for i, phase in enumerate(phases):
        x = start_x + i*(box_w + spacing)
        # Box
        dwg.add(dwg.rect(insert=(x, pipeline_y), size=(box_w, box_h), rx=4, fill=grey_bg, stroke=dark_blue, stroke_width=2))
        # Title
        dwg.add(dwg.text(phase["title"], insert=(x + box_w/2, pipeline_y + 30), font_family=font_family, font_size=14, font_weight="bold", fill=dark_blue, text_anchor="middle"))
        # Desc
        for j, line in enumerate(phase["desc"].split("\n")):
            dwg.add(dwg.text(line, insert=(x + box_w/2, pipeline_y + 55 + j*20), font_family=font_family, font_size=12, fill=dark_grey, text_anchor="middle"))
        
        # Arrow to next
        if i < 3:
            ax = x + box_w
            ay = pipeline_y + box_h/2
            # line and arrowhead
            dwg.add(dwg.line(start=(ax, ay), end=(ax+spacing-4, ay), stroke=dark_grey, stroke_width=2))
            dwg.add(dwg.polygon(points=[(ax+spacing, ay), (ax+spacing-6, ay-4), (ax+spacing-6, ay+4)], fill=dark_grey))
    
    # Pipeline label
    dwg.add(dwg.text("Synchronous Four-Phase Handshake Pipeline", insert=(width/2, pipeline_y - 15), font_family=font_family, font_size=16, font_weight="bold", fill=dark_grey, text_anchor="middle"))

    # -----------------------------
    # Bottom Left: FSM
    # -----------------------------
    fsm_x = 50
    fsm_y = 220
    fsm_w = 400
    fsm_h = 350
    dwg.add(dwg.text("Discrete Authority State Machine", insert=(fsm_x + fsm_w/2, fsm_y + 20), font_family=font_family, font_size=16, font_weight="bold", fill=dark_grey, text_anchor="middle"))
    
    # State boxes
    states = {
        "Auto": {"pos": (fsm_x + 40, fsm_y + 60), "label": "Autonomous"},
        "Pend": {"pos": (fsm_x + 260, fsm_y + 60), "label": "Handover-Pending"},
        "Blend": {"pos": (fsm_x + 260, fsm_y + 160), "label": "Bumpless-Blend"},
        "Manual": {"pos": (fsm_x + 260, fsm_y + 260), "label": "Human-Manual"},
        "Degrad": {"pos": (fsm_x + 40, fsm_y + 260), "label": "Degraded-Fallback\n(Safe Stop)"}
    }
    
    state_w, state_h = 130, 60
    
    def draw_arrow(start, end, label, color=dark_grey, curve=False, cp_offset=(0,0), t_offset=(0,-10)):
        sx, sy = start
        ex, ey = end
        if not curve:
            dwg.add(dwg.line(start=(sx, sy), end=(ex, ey), stroke=color, stroke_width=2))
            angle = __import__('math').atan2(ey-sy, ex-sx)
            dwg.add(dwg.polygon(points=[(ex, ey), 
                                        (ex - 8*__import__('math').cos(angle-0.5), ey - 8*__import__('math').sin(angle-0.5)), 
                                        (ex - 8*__import__('math').cos(angle+0.5), ey - 8*__import__('math').sin(angle+0.5))], fill=color))
        else:
            path = dwg.path(d=f"M {sx},{sy} Q {sx+cp_offset[0]},{sy+cp_offset[1]} {ex},{ey}", fill="none", stroke=color, stroke_width=2)
            dwg.add(path)
            angle = __import__('math').atan2(ey - (sy+cp_offset[1]), ex - (sx+cp_offset[0]))
            dwg.add(dwg.polygon(points=[(ex, ey), 
                                        (ex - 8*__import__('math').cos(angle-0.5), ey - 8*__import__('math').sin(angle-0.5)), 
                                        (ex - 8*__import__('math').cos(angle+0.5), ey - 8*__import__('math').sin(angle+0.5))], fill=color))
            
        mid_x = (sx + ex)/2 if not curve else sx + cp_offset[0]/2
        mid_y = (sy + ey)/2 if not curve else sy + cp_offset[1]/2
        
        for i, l in enumerate(label.split('\n')):
            dwg.add(dwg.text(l, insert=(mid_x + t_offset[0], mid_y + t_offset[1] + i*15), font_family=font_family, font_size=10, fill=color, text_anchor="middle", filter="url(#solid_bg)"))

    # Create filter for background
    filter = dwg.defs.add(dwg.filter(id="solid_bg", x="0", y="0", width="1", height="1"))
    filter.feFlood(flood_color="#FFFFFF", result="bg")
    filter.feMerge(["bg", "SourceGraphic"])
            
    # Transitions
    s_c = lambda k: (states[k]["pos"][0] + state_w/2, states[k]["pos"][1] + state_h/2)
    draw_arrow(s_c("Auto"), s_c("Pend"), "Request", dark_blue, t_offset=(0, -8))
    draw_arrow(s_c("Pend"), s_c("Blend"), "ACK + Auth", dark_blue, t_offset=(30, 0))
    draw_arrow(s_c("Blend"), s_c("Manual"), "Commit\nα=1.0", dark_blue, t_offset=(30, -5))
    
    # Timeouts / Fallbacks
    draw_arrow((states["Pend"]["pos"][0], states["Pend"]["pos"][1]+state_h/2), 
               (states["Degrad"]["pos"][0]+state_w/2, states["Degrad"]["pos"][1]), 
               "Timeout\n500ms", deep_red, t_offset=(20, -10))
    draw_arrow((states["Manual"]["pos"][0], states["Manual"]["pos"][1]+state_h/2), 
               (states["Degrad"]["pos"][0]+state_w, states["Degrad"]["pos"][1]+state_h/2), 
               "Lease Expire", deep_red, t_offset=(0, -8))
    draw_arrow((states["Auto"]["pos"][0]+state_w/2, states["Auto"]["pos"][1]+state_h), 
               (states["Degrad"]["pos"][0]+state_w/2, states["Degrad"]["pos"][1]), 
               "Critical\nFault", deep_red, t_offset=(-30, -10))
               
    # Draw boxes over arrows
    for k, v in states.items():
        box_c = deep_red if k == "Degrad" else dark_blue
        dwg.add(dwg.rect(insert=v["pos"], size=(state_w, state_h), rx=4, fill=grey_bg, stroke=box_c, stroke_width=2))
        for j, line in enumerate(v["label"].split('\n')):
            dwg.add(dwg.text(line, insert=(v["pos"][0] + state_w/2, v["pos"][1] + 35 + j*15), font_family=font_family, font_size=12, font_weight="bold", fill=dark_grey, text_anchor="middle"))

    # -----------------------------
    # Bottom Right: Arbitration Hierarchy
    # -----------------------------
    arb_x = 520
    arb_y = 220
    arb_w = 340
    arb_h = 350
    dwg.add(dwg.text("Real-Time Arbitration Hierarchy", insert=(arb_x + arb_w/2, arb_y + 20), font_family=font_family, font_size=16, font_weight="bold", fill=dark_grey, text_anchor="middle"))
    
    tiers = [
        {"label": "Tier 1: Safety Enforcer", "sub": "h(x) ≥ 0, Kinematic Bounds", "color": deep_red},
        {"label": "Tier 2: Human Manual", "sub": "α = 1.0, Lease active", "color": dark_blue},
        {"label": "Tier 3: Learned Policy", "sub": "Nominal Autonomous", "color": stroke_grey}
    ]
    
    tier_w = 280
    tier_h = 60
    start_ty = arb_y + 60
    
    for i, tier in enumerate(tiers):
        tx = arb_x + (arb_w - tier_w)/2
        ty = start_ty + i*(tier_h + 15)
        dwg.add(dwg.rect(insert=(tx, ty), size=(tier_w, tier_h), rx=4, fill=grey_bg, stroke=tier["color"], stroke_width=2))
        # Arrow pointing down (preemption)
        if i > 0:
            dwg.add(dwg.line(start=(tx + tier_w/2, ty - 15), end=(tx + tier_w/2, ty - 2), stroke=dark_grey, stroke_width=2))
            dwg.add(dwg.polygon(points=[(tx + tier_w/2, ty), (tx + tier_w/2 - 4, ty - 6), (tx + tier_w/2 + 4, ty - 6)], fill=dark_grey))
            
        # Label
        dwg.add(dwg.text(tier["label"], insert=(tx + 10, ty + 25), font_family=font_family, font_size=14, font_weight="bold", fill=dark_grey))
        dwg.add(dwg.text(tier["sub"], insert=(tx + 10, ty + 45), font_family=font_family, font_size=12, fill=dark_grey))

    # Evidence audit tuple
    audit_y = start_ty + 3 * (tier_h + 15) + 10
    dwg.add(dwg.rect(insert=(arb_x + (arb_w - tier_w)/2, audit_y), size=(tier_w, 50), rx=4, fill="#EDF2F7", stroke=dark_grey, stroke_width=1, stroke_dasharray="4,4"))
    dwg.add(dwg.text("Evidence Audit Tuple (64-byte)", insert=(arb_x + arb_w/2, audit_y + 20), font_family=font_family, font_size=12, font_weight="bold", fill=dark_grey, text_anchor="middle"))
    dwg.add(dwg.text("IEEE 1588 ts | S_auth | x | u_act | SHA-256 HMAC", insert=(arb_x + arb_w/2, audit_y + 40), font_family=font_family, font_size=11, fill=dark_grey, text_anchor="middle"))

    dwg.save()

if __name__ == '__main__':
    draw_fsm_fig()

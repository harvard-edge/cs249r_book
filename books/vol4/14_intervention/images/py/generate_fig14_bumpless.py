import svgwrite
import math

def draw_bumpless_fig():
    width, height = 900, 450
    dwg = svgwrite.Drawing('books/vol4/14_intervention/images/svg/fig14_bumpless_transfer_dynamics.svg', size=(width, height))

    bg = "#FFFFFF"
    dark_blue = "#1F407A"
    deep_red = "#A51C30"
    grey_bg = "#F7FAFC"
    dark_grey = "#2D3748"
    grid_color = "#E2E8F0"
    
    dwg.add(dwg.rect(insert=(0,0), size=(width, height), fill=bg))
    font_family = "system-ui, -apple-system, sans-serif"

    # We have two panels: A (Left, Step Discontinuity), B (Right, C^2 Bumpless Blending)
    panel_w = 400
    panel_h = 350
    pad_y = 50
    
    def draw_panel(dwg, x_start, title, type="step"):
        dwg.add(dwg.text(title, insert=(x_start + panel_w/2, pad_y - 15), font_family=font_family, font_size=16, font_weight="bold", fill=dark_grey, text_anchor="middle"))
        
        # Axes
        origin_x = x_start + 40
        origin_y = pad_y + panel_h - 40
        graph_w = panel_w - 60
        graph_h = panel_h - 60
        
        # Grid
        for i in range(5):
            y = origin_y - i * (graph_h / 4)
            dwg.add(dwg.line(start=(origin_x, y), end=(origin_x + graph_w, y), stroke=grid_color, stroke_width=1))
        for i in range(5):
            x = origin_x + i * (graph_w / 4)
            dwg.add(dwg.line(start=(x, origin_y), end=(x, origin_y - graph_h), stroke=grid_color, stroke_width=1))
            
        dwg.add(dwg.line(start=(origin_x, origin_y), end=(origin_x + graph_w, origin_y), stroke=dark_grey, stroke_width=2))
        dwg.add(dwg.line(start=(origin_x, origin_y), end=(origin_x, origin_y - graph_h), stroke=dark_grey, stroke_width=2))
        
        dwg.add(dwg.text("Time (ms)", insert=(origin_x + graph_w/2, origin_y + 25), font_family=font_family, font_size=12, fill=dark_grey, text_anchor="middle"))
        
        t0 = graph_w * 0.2
        t1 = graph_w * (0.2 if type == "step" else 0.6)
        
        # Draw Torque (Blue)
        path = f"M {origin_x}, {origin_y - 20} "
        if type == "step":
            path += f"L {origin_x + t0}, {origin_y - 20} "
            path += f"L {origin_x + t0}, {origin_y - graph_h + 20} "
            path += f"L {origin_x + graph_w}, {origin_y - graph_h + 20}"
        else:
            path += f"L {origin_x + t0}, {origin_y - 20} "
            # Sigmoid C2 curve
            pts = []
            steps = 50
            for i in range(steps+1):
                t_ratio = i/steps
                # quintic ease in out: 6t^5 - 15t^4 + 10t^3
                alpha = 6*(t_ratio**5) - 15*(t_ratio**4) + 10*(t_ratio**3)
                px = origin_x + t0 + t_ratio * (t1 - t0)
                py = (origin_y - 20) - alpha * (graph_h - 40)
                pts.append(f"{px},{py}")
            path += " L " + " L ".join(pts)
            path += f" L {origin_x + graph_w}, {origin_y - graph_h + 20}"
            
        dwg.add(dwg.path(d=path, fill="none", stroke=dark_blue, stroke_width=3))
        
        # Jerk (Red)
        if type == "step":
            jerk_path = f"M {origin_x}, {origin_y} L {origin_x + t0 - 2}, {origin_y} "
            jerk_path += f"L {origin_x + t0}, {origin_y - graph_h + 10} "
            jerk_path += f"L {origin_x + t0 + 2}, {origin_y} L {origin_x + graph_w}, {origin_y}"
        else:
            jerk_path = f"M {origin_x}, {origin_y} L {origin_x + t0}, {origin_y} "
            pts = []
            steps = 50
            for i in range(steps+1):
                t_ratio = i/steps
                # derivative of quintic: 30t^4 - 60t^3 + 30t^2
                j_mag = 30*(t_ratio**4) - 60*(t_ratio**3) + 30*(t_ratio**2)
                px = origin_x + t0 + t_ratio * (t1 - t0)
                py = origin_y - j_mag * 50
                pts.append(f"{px},{py}")
            jerk_path += " L " + " L ".join(pts)
            jerk_path += f" L {origin_x + graph_w}, {origin_y}"
            
        dwg.add(dwg.path(d=jerk_path, fill="none", stroke=deep_red, stroke_width=2))
        
        # Labels for traces
        dwg.add(dwg.text("Commanded Torque τ(t)", insert=(origin_x + 10, origin_y - graph_h + 10), font_family=font_family, font_size=12, font_weight="bold", fill=dark_blue))
        if type == "step":
            dwg.add(dwg.text("Infinite Jerk Spike", insert=(origin_x + t0 + 10, origin_y - graph_h/2), font_family=font_family, font_size=12, font_weight="bold", fill=deep_red))
            # Ringing
            ring_path = f"M {origin_x + t0}, {origin_y - graph_h + 20} "
            for i in range(60):
                rx = origin_x + t0 + i*3
                ry = (origin_y - graph_h + 20) - math.exp(-i/15) * 30 * math.cos(i*0.8)
                ring_path += f"L {rx},{ry} "
            dwg.add(dwg.path(d=ring_path, fill="none", stroke=dark_grey, stroke_width=1.5, stroke_dasharray="2,2"))
            dwg.add(dwg.text("25Hz Drivetrain Ringing", insert=(origin_x + t0 + 40, origin_y - graph_h + 40), font_family=font_family, font_size=11, fill=dark_grey))
        else:
            dwg.add(dwg.text("Bounded Jerk j(t)", insert=(origin_x + (t0+t1)/2 - 10, origin_y - 60), font_family=font_family, font_size=12, font_weight="bold", fill=deep_red))
            
    draw_panel(dwg, 30, "Panel A: Unblended Step (0ms)", "step")
    draw_panel(dwg, 470, "Panel B: C² Quintic Blending (750ms)", "blend")

    dwg.save()

if __name__ == '__main__':
    draw_bumpless_fig()

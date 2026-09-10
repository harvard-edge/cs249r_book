import os
import math

def generate_svg():
    width = 800
    height = 500
    
    # Colors
    c_dark_blue = "#1F407A"
    c_grey_light = "#E2E8F0"
    c_grey_med = "#CBD5E0"
    c_grey_dark = "#2D3748"
    c_red = "#A51C30"
    c_valid = "#38A169" # green
    c_degraded = "#D69E2E" # yellow
    c_bg = "#FFFFFF"

    svg = [
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg" style="background-color: {c_bg}; font-family: sans-serif;">',
        '  <defs>',
        '    <marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">',
        f'      <path d="M0,0 L0,6 L9,3 z" fill="{c_grey_dark}" />',
        '    </marker>',
        '  </defs>',
    ]
    
    # Chart area
    margin_left = 80
    margin_bottom = 100
    plot_width = 650
    plot_height = 300
    
    x_origin = margin_left
    y_origin = height - margin_bottom
    
    # Data range
    t_max = 120 # ms
    e_max_plot = 2.0 # mm
    
    def get_xy(t_ms, e_mm):
        x = x_origin + (t_ms / t_max) * plot_width
        y = y_origin - (e_mm / e_max_plot) * plot_height
        return x, y

    # Draw grid and axes
    svg.append(f'  <!-- Grid -->')
    for e in [0.5, 1.0, 1.5, 2.0]:
        _, y = get_xy(0, e)
        svg.append(f'  <line x1="{x_origin}" y1="{y}" x2="{x_origin+plot_width}" y2="{y}" stroke="{c_grey_light}" stroke-width="1"/>')
        svg.append(f'  <text x="{x_origin-10}" y="{y+5}" font-size="12" fill="{c_grey_dark}" text-anchor="end">{e:.1f}</text>')
        
    for t in [20, 40, 60, 80, 100, 120]:
        x, _ = get_xy(t, 0)
        svg.append(f'  <line x1="{x}" y1="{y_origin}" x2="{x}" y2="{y_origin-plot_height}" stroke="{c_grey_light}" stroke-width="1"/>')
        svg.append(f'  <text x="{x}" y="{y_origin+20}" font-size="12" fill="{c_grey_dark}" text-anchor="middle">{t}</text>')
        
    # Axes
    svg.append(f'  <line x1="{x_origin}" y1="{y_origin}" x2="{x_origin+plot_width}" y2="{y_origin}" stroke="{c_grey_dark}" stroke-width="2"/>')
    svg.append(f'  <line x1="{x_origin}" y1="{y_origin}" x2="{x_origin}" y2="{y_origin-plot_height}" stroke="{c_grey_dark}" stroke-width="2"/>')
    svg.append(f'  <text x="{x_origin+plot_width/2}" y="{y_origin+40}" font-size="14" fill="{c_grey_dark}" font-weight="bold" text-anchor="middle">Time Since Last Measurement (ms)</text>')
    svg.append(f'  <text x="{x_origin-50}" y="{y_origin-plot_height/2}" font-size="14" fill="{c_grey_dark}" font-weight="bold" text-anchor="middle" transform="rotate(-90 {x_origin-50} {y_origin-plot_height/2})">Spatial Error Bound E(t) (mm)</text>')
    
    # Task clearance limit
    limit_y = get_xy(0, 1.0)[1]
    svg.append(f'  <line x1="{x_origin}" y1="{limit_y}" x2="{x_origin+plot_width}" y2="{limit_y}" stroke="{c_red}" stroke-width="2" stroke-dasharray="8,4"/>')
    svg.append(f'  <text x="{x_origin+plot_width-10}" y="{limit_y-10}" font-size="14" fill="{c_red}" font-weight="bold" text-anchor="end">Task Clearance (E_max = 1.0 mm)</text>')
    
    # Draw curve E(t) = 0.1 + 12t + 25t^2
    path_points = []
    fill_points = [(x_origin, y_origin)]
    for t_ms in range(0, t_max + 1, 2):
        t_sec = t_ms / 1000.0
        e_mm = 0.1 + 12.0 * t_sec + 25.0 * (t_sec**2)
        x, y = get_xy(t_ms, e_mm)
        if e_mm <= e_max_plot:
            path_points.append(f"{x},{y}")
            fill_points.append((x, y))
            
    fill_points.append((fill_points[-1][0], y_origin))
    fill_path = " ".join([f"{x},{y}" for x, y in fill_points])
    
    # Shaded area under curve
    svg.append(f'  <polygon points="{fill_path}" fill="{c_dark_blue}" fill-opacity="0.1"/>')
    # The curve itself
    svg.append(f'  <polyline points="{" ".join(path_points)}" fill="none" stroke="{c_dark_blue}" stroke-width="3"/>')
    
    # Equation label
    svg.append(f'  <text x="{x_origin+50}" y="{y_origin-150}" font-size="14" fill="{c_dark_blue}" font-weight="bold">E(t) = E(0) + 3σ_v t + 1/2 a_dist t²</text>')
    
    # Invalidation horizon
    t_exp = 66
    x_exp, _ = get_xy(t_exp, 0)
    svg.append(f'  <line x1="{x_exp}" y1="{y_origin}" x2="{x_exp}" y2="{limit_y}" stroke="{c_red}" stroke-width="2"/>')
    svg.append(f'  <circle cx="{x_exp}" cy="{limit_y}" r="6" fill="{c_red}"/>')
    
    svg.append(f'  <text x="{x_exp+10}" y="{y_origin-40}" font-size="14" fill="{c_red}" font-weight="bold">t_exp = 66 ms</text>')
    svg.append(f'  <text x="{x_exp+10}" y="{y_origin-20}" font-size="12" fill="{c_red}">(Invalidation Horizon)</text>')
    
    # Lifecycle Bar below axis
    bar_y = y_origin + 60
    bar_h = 25
    
    # Valid (0 to 33ms)
    t_deg = 33
    x_deg, _ = get_xy(t_deg, 0)
    svg.append(f'  <rect x="{x_origin}" y="{bar_y}" width="{x_deg-x_origin}" height="{bar_h}" fill="{c_valid}" rx="2"/>')
    svg.append(f'  <text x="{x_origin + (x_deg-x_origin)/2}" y="{bar_y+17}" font-size="12" fill="#FFFFFF" font-weight="bold" text-anchor="middle">VALID</text>')
    
    # Degraded (33ms to 66ms)
    svg.append(f'  <rect x="{x_deg}" y="{bar_y}" width="{x_exp-x_deg}" height="{bar_h}" fill="{c_degraded}" rx="2"/>')
    svg.append(f'  <text x="{x_deg + (x_exp-x_deg)/2}" y="{bar_y+17}" font-size="12" fill="#FFFFFF" font-weight="bold" text-anchor="middle">DEGRADED</text>')
    
    # Expired (66ms+)
    x_end = x_origin + plot_width
    svg.append(f'  <rect x="{x_exp}" y="{bar_y}" width="{x_end-x_exp}" height="{bar_h}" fill="{c_red}" rx="2"/>')
    svg.append(f'  <text x="{x_exp + (x_end-x_exp)/2}" y="{bar_y+17}" font-size="12" fill="#FFFFFF" font-weight="bold" text-anchor="middle">EXPIRED (Safe Stop)</text>')
    
    svg.append('</svg>')
    
    os.makedirs(os.path.dirname('/Users/VJ/GitHub/MLSysBook-vol4-physical/books/vol4/memory/images/svg/fig09_uncertainty_growth.svg'), exist_ok=True)
    with open('/Users/VJ/GitHub/MLSysBook-vol4-physical/books/vol4/memory/images/svg/fig09_uncertainty_growth.svg', 'w') as f:
        f.write('\\n'.join(svg))
    print("fig09_uncertainty_growth.svg generated")

if __name__ == "__main__":
    generate_svg()

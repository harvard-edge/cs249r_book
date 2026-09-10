import math

def generate_ray_frustum():
    svg_width = 1000
    svg_height = 500
    
    c_blue = "#1F407A"
    c_gray_light = "#E2E8F0"
    c_gray_dark = "#2D3748"
    c_red = "#A51C30"
    c_white = "#FFFFFF"

    svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {svg_width} {svg_height}" width="{svg_width}" height="{svg_height}">\n'
    
    # Styles
    svg += f'''
    <defs>
        <marker id="arrow_blue" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="{c_blue}" />
        </marker>
        <marker id="arrow_gray" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="{c_gray_dark}" />
        </marker>
    </defs>
    '''
    
    svg += f'<rect width="100%" height="100%" fill="{c_white}"/>\n'

    # Title or section labels
    sections = [
        {"title": "(1) Hardware Ingress &amp; Patchification", "x": 150},
        {"title": "(2) 2D-to-3D Ray Lifting", "x": 450},
        {"title": "(3) Metric Voxel Splatting", "x": 700},
        {"title": "(4) Propagated Uncertainty", "x": 900}
    ]
    for sec in sections:
        svg += f'<text x="{sec["x"]}" y="40" font-family="sans-serif" font-weight="bold" font-size="14" fill="{c_blue}" text-anchor="middle">{sec["title"]}</text>\n'

    # --- Section 1: Ingress ---
    # Draw a 2D image plane with some patches
    img_x = 100
    img_y = 150
    img_w = 100
    img_h = 100
    svg += f'<rect x="{img_x-img_w/2}" y="{img_y-img_h/2}" width="{img_w}" height="{img_h}" fill="none" stroke="{c_gray_dark}" stroke-width="2"/>\n'
    # draw grid
    for i in range(1, 4):
        y_line = img_y - img_h/2 + i * img_h/4
        svg += f'<line x1="{img_x-img_w/2}" y1="{y_line}" x2="{img_x+img_w/2}" y2="{y_line}" stroke="{c_gray_light}" stroke-width="1"/>\n'
        x_line = img_x - img_w/2 + i * img_w/4
        svg += f'<line x1="{x_line}" y1="{img_y-img_h/2}" x2="{x_line}" y2="{img_y+img_h/2}" stroke="{c_gray_light}" stroke-width="1"/>\n'
    
    # Highlight a single patch
    patch_w = img_w/4
    patch_h = img_h/4
    svg += f'<rect x="{img_x-img_w/2 + patch_w}" y="{img_y-img_h/2 + patch_h}" width="{patch_w}" height="{patch_h}" fill="{c_blue}" fill-opacity="0.2" stroke="{c_blue}" stroke-width="2"/>\n'
    svg += f'<text x="{img_x}" y="{img_y + img_h/2 + 20}" font-family="sans-serif" font-size="12" fill="{c_gray_dark}" text-anchor="middle">2D Pixel Array</text>\n'

    # Arrow to Ray Lifting
    svg += f'<path d="M {img_x + img_w/2 + 10} {img_y} L {280} {img_y}" stroke="{c_gray_dark}" stroke-width="2" marker-end="url(#arrow_gray)" fill="none"/>\n'

    # --- Section 2: Ray Lifting ---
    # Camera origin
    cam_x = 300
    cam_y = 150
    svg += f'<circle cx="{cam_x}" cy="{cam_y}" r="4" fill="{c_gray_dark}"/>\n'
    svg += f'<text x="{cam_x}" y="{cam_y + 20}" font-family="sans-serif" font-size="12" fill="{c_gray_dark}" text-anchor="middle">Camera Center</text>\n'

    # Rays
    angle1 = -15
    angle2 = 15
    ray_len = 250
    for ang in [angle1, 0, angle2]:
        rad = math.radians(ang)
        ex = cam_x + ray_len * math.cos(rad)
        ey = cam_y + ray_len * math.sin(rad)
        svg += f'<line x1="{cam_x}" y1="{cam_y}" x2="{ex}" y2="{ey}" stroke="{c_gray_light}" stroke-width="2" stroke-dasharray="4,4"/>\n'

    # Discretized depth bins along central ray
    bin_centers = [cam_x + 80, cam_x + 130, cam_x + 180, cam_x + 230]
    for bx in bin_centers:
        svg += f'<line x1="{bx}" y1="{cam_y - 20}" x2="{bx}" y2="{cam_y + 20}" stroke="{c_blue}" stroke-width="2"/>\n'
    
    svg += f'<text x="{cam_x + 150}" y="{cam_y - 30}" font-family="sans-serif" font-size="12" fill="{c_blue}" text-anchor="middle">Discrete Depth Bins</text>\n'

    # --- Section 3: Voxel Splatting ---
    vox_x = 620
    vox_y = 100
    vox_w = 160
    vox_h = 100
    
    # Arrow to Voxel Splatting
    svg += f'<path d="M {cam_x + ray_len + 10} {cam_y} L {vox_x - 20} {cam_y}" stroke="{c_gray_dark}" stroke-width="2" marker-end="url(#arrow_gray)" fill="none"/>\n'
    
    # 2D Grid representing BEV Voxels
    svg += f'<rect x="{vox_x}" y="{vox_y}" width="{vox_w}" height="{vox_h}" fill="none" stroke="{c_gray_dark}" stroke-width="2"/>\n'
    for i in range(1, 8):
        x_line = vox_x + i * 20
        svg += f'<line x1="{x_line}" y1="{vox_y}" x2="{x_line}" y2="{vox_y+vox_h}" stroke="{c_gray_light}" stroke-width="1"/>\n'
    for i in range(1, 5):
        y_line = vox_y + i * 20
        svg += f'<line x1="{vox_x}" y1="{y_line}" x2="{vox_x+vox_w}" y2="{y_line}" stroke="{c_gray_light}" stroke-width="1"/>\n'

    # Splatting effect
    splat_xs = [vox_x + 40, vox_x + 80, vox_x + 120]
    splat_y = vox_y + 40
    for sx in splat_xs:
        svg += f'<rect x="{sx}" y="{splat_y}" width="20" height="20" fill="{c_blue}" fill-opacity="0.6"/>\n'
    
    svg += f'<text x="{vox_x + vox_w/2}" y="{vox_y + vox_h + 20}" font-family="sans-serif" font-size="12" fill="{c_gray_dark}" text-anchor="middle">Metric Voxel Grid</text>\n'

    # --- Section 4: Propagated Uncertainty ---
    # Arrow
    svg += f'<path d="M {vox_x + vox_w + 10} {cam_y} L {840} {cam_y}" stroke="{c_gray_dark}" stroke-width="2" marker-end="url(#arrow_gray)" fill="none"/>\n'
    
    unc_x = 880
    unc_y = 150
    # Center object
    svg += f'<rect x="{unc_x-5}" y="{unc_y-5}" width="10" height="10" fill="{c_gray_dark}"/>\n'
    # Ellipsoids elongating with range. (Range is assumed along x axis here)
    # The farther, the more elongated. 
    # Wait, the drawing just shows the final output. We can draw the object and a large covariance ellipse.
    svg += f'<ellipse cx="{unc_x}" cy="{unc_y}" rx="40" ry="15" fill="none" stroke="{c_red}" stroke-width="2" stroke-dasharray="2,2"/>\n'
    svg += f'<text x="{unc_x}" y="{unc_y + 35}" font-family="sans-serif" font-size="12" fill="{c_red}" text-anchor="middle">Elongated Covariance</text>\n'
    svg += f'<text x="{unc_x}" y="{unc_y + 50}" font-family="sans-serif" font-size="12" fill="{c_red}" text-anchor="middle">at Range</text>\n'
    
    # Equation
    eq_y = 350
    svg += f'<rect x="150" y="{eq_y-40}" width="700" height="80" rx="4" fill="{c_white}" stroke="{c_gray_dark}" stroke-width="1" stroke-dasharray="4,4"/>\n'
    svg += f'<text x="500" y="{eq_y-15}" font-family="sans-serif" font-weight="bold" font-size="14" fill="{c_gray_dark}" text-anchor="middle">Clearance Contract</text>\n'
    
    eq_text = 'Measurement must include: (1) Timestamp t<tspan baseline-shift="sub" font-size="10">0</tspan>, (2) Frame F<tspan baseline-shift="sub" font-size="10">sensor</tspan>, (3) Covariance &#931;<tspan baseline-shift="sub" font-size="10">z</tspan>'
    
    svg += f'<text x="500" y="{eq_y+15}" font-family="sans-serif" font-size="14" fill="{c_blue}" text-anchor="middle">{eq_text}</text>\n'

    svg += '</svg>\n'

    with open('/Users/VJ/GitHub/MLSysBook-vol4-physical/books/vol4/perception/images/svg/fig08_ray_frustum_voxel_lifting.svg', 'w') as f:
        f.write(svg)

if __name__ == "__main__":
    generate_ray_frustum()

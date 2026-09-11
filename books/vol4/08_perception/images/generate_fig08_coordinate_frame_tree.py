import math

def generate_coordinate_tree():
    svg_width = 900
    svg_height = 450
    
    # Colors
    c_blue = "#1F407A"
    c_gray_light = "#E2E8F0"
    c_gray_dark = "#2D3748"
    c_red = "#A51C30"
    c_white = "#FFFFFF"

    # SVG header
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {svg_width} {svg_height}" width="{svg_width}" height="{svg_height}">\n'
    
    # Styles
    svg += f'''
    <defs>
        <marker id="arrow_blue" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="{c_blue}" />
        </marker>
        <marker id="arrow_red" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="{c_red}" />
        </marker>
        <marker id="arrow_gray" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="{c_gray_dark}" />
        </marker>
    </defs>
    '''
    
    # Background
    svg += f'<rect width="100%" height="100%" fill="{c_white}"/>\n'
    
    # Frames definition
    # We lay them out left to right: earth -> map -> odom -> body -> sensor
    
    frames = [
        {"id": "earth", "label": "F_earth", "desc": "Root Datum", "x": 100, "y": 200},
        {"id": "map", "label": "F_map", "desc": "Global Map", "x": 280, "y": 200},
        {"id": "odom", "label": "F_odom", "desc": "Local Odometry", "x": 460, "y": 200},
        {"id": "body", "label": "F_body", "desc": "base_link", "x": 640, "y": 200},
        {"id": "sensor", "label": "F_sensor", "desc": "sensor_link", "x": 820, "y": 200},
    ]
    
    box_w = 120
    box_h = 60
    
    # Edges
    # earth to map
    svg += f'<path d="M {frames[0]["x"]+box_w/2} {frames[0]["y"]} L {frames[1]["x"]-box_w/2} {frames[1]["y"]}" stroke="{c_gray_dark}" stroke-width="2" marker-end="url(#arrow_gray)" fill="none" stroke-dasharray="4,4"/>\n'
    svg += f'<text x="{(frames[0]["x"]+frames[1]["x"])/2}" y="{frames[0]["y"]-15}" font-family="sans-serif" font-size="12" fill="{c_gray_dark}" text-anchor="middle">Piecewise C0</text>\n'
    
    # map to odom (discontinuous, red)
    svg += f'<path d="M {frames[1]["x"]+box_w/2} {frames[1]["y"]} L {frames[2]["x"]-box_w/2} {frames[2]["y"]}" stroke="{c_red}" stroke-width="2" marker-end="url(#arrow_red)" fill="none" />\n'
    svg += f'<text x="{(frames[1]["x"]+frames[2]["x"])/2}" y="{frames[1]["y"]-25}" font-family="sans-serif" font-size="12" font-weight="bold" fill="{c_red}" text-anchor="middle">Discontinuous</text>\n'
    svg += f'<text x="{(frames[1]["x"]+frames[2]["x"])/2}" y="{frames[1]["y"]-10}" font-family="sans-serif" font-size="12" fill="{c_red}" text-anchor="middle">Map Corrections</text>\n'
    
    # odom to body (continuous, blue)
    svg += f'<path d="M {frames[2]["x"]+box_w/2} {frames[2]["y"]} L {frames[3]["x"]-box_w/2} {frames[3]["y"]}" stroke="{c_blue}" stroke-width="3" marker-end="url(#arrow_blue)" fill="none" />\n'
    svg += f'<text x="{(frames[2]["x"]+frames[3]["x"])/2}" y="{frames[2]["y"]-25}" font-family="sans-serif" font-size="12" font-weight="bold" fill="{c_blue}" text-anchor="middle">Continuous (C1)</text>\n'
    svg += f'<text x="{(frames[2]["x"]+frames[3]["x"])/2}" y="{frames[2]["y"]-10}" font-family="sans-serif" font-size="12" fill="{c_blue}" text-anchor="middle">Dead Reckoning</text>\n'
    
    # body to sensor (rigid, gray)
    svg += f'<path d="M {frames[3]["x"]+box_w/2} {frames[3]["y"]} L {frames[4]["x"]-box_w/2} {frames[4]["y"]}" stroke="{c_gray_dark}" stroke-width="2" marker-end="url(#arrow_gray)" fill="none" />\n'
    svg += f'<text x="{(frames[3]["x"]+frames[4]["x"])/2}" y="{frames[3]["y"]-25}" font-family="sans-serif" font-size="12" font-weight="bold" fill="{c_gray_dark}" text-anchor="middle">Rigid Extrinsic</text>\n'
    svg += f'<text x="{(frames[3]["x"]+frames[4]["x"])/2}" y="{frames[3]["y"]-10}" font-family="sans-serif" font-size="12" fill="{c_gray_dark}" text-anchor="middle">Mounting</text>\n'
    
    # Draw boxes
    for f in frames:
        svg += f'<rect x="{f["x"]-box_w/2}" y="{f["y"]-box_h/2}" width="{box_w}" height="{box_h}" fill="{c_gray_light}" stroke="{c_blue}" stroke-width="2"/>\n'
        # Label (e.g. F_earth)
        label_text = f['label'].replace('F_', 'F<tspan baseline-shift="sub" font-size="10">') + '</tspan>'
        svg += f'<text x="{f["x"]}" y="{f["y"]}" font-family="sans-serif" font-weight="bold" font-size="16" fill="{c_blue}" text-anchor="middle">{label_text}</text>\n'
        svg += f'<text x="{f["x"]}" y="{f["y"]+18}" font-family="sans-serif" font-size="12" fill="{c_gray_dark}" text-anchor="middle">{f["desc"]}</text>\n'

    # Add the Covariance Equation Box at the bottom
    eq_y = 350
    svg += f'<rect x="100" y="{eq_y-40}" width="700" height="80" fill="{c_white}" stroke="{c_gray_dark}" stroke-width="1" stroke-dasharray="4,4"/>\n'
    svg += f'<text x="450" y="{eq_y-15}" font-family="sans-serif" font-weight="bold" font-size="14" fill="{c_gray_dark}" text-anchor="middle">Spatial Covariance Composition</text>\n'
    
    # Sigma_map approx J_odom Sigma_drift J_odom^T + R_body Sigma_ext R_body^T + R_tot Sigma_z R_tot^T
    # Using simple text approximation
    eq_text = '&#931;<tspan baseline-shift="sub" font-size="10">map</tspan> &#8776; '
    eq_text += 'J<tspan baseline-shift="sub" font-size="10">odom</tspan> &#931;<tspan baseline-shift="sub" font-size="10">drift</tspan> J<tspan baseline-shift="sub" font-size="10">odom</tspan><tspan baseline-shift="super" font-size="10">T</tspan> + '
    eq_text += 'R<tspan baseline-shift="sub" font-size="10">body</tspan> &#931;<tspan baseline-shift="sub" font-size="10">ext</tspan> R<tspan baseline-shift="sub" font-size="10">body</tspan><tspan baseline-shift="super" font-size="10">T</tspan> + '
    eq_text += 'R<tspan baseline-shift="sub" font-size="10">tot</tspan> &#931;<tspan baseline-shift="sub" font-size="10">z</tspan> R<tspan baseline-shift="sub" font-size="10">tot</tspan><tspan baseline-shift="super" font-size="10">T</tspan>'
    
    svg += f'<text x="450" y="{eq_y+15}" font-family="serif" font-style="italic" font-size="16" fill="{c_blue}" text-anchor="middle">{eq_text}</text>\n'

    # Close SVG
    svg += '</svg>\n'
    
    with open('/Users/VJ/GitHub/MLSysBook-vol4-physical/books/vol4/08_perception/images/svg/fig08_coordinate_frame_tree.svg', 'w') as f:
        f.write(svg)

if __name__ == "__main__":
    generate_coordinate_tree()

import os
import glob
import re
from lxml import etree as ET

COLOR_MAP = {
    # Compute (blue)
    '#e8f2fc': '#cfe2f3', '#eef4fb': '#cfe2f3', '#f0f6ff': '#cfe2f3', '#f0f8ff': '#cfe2f3', '#f7f9fc': '#cfe2f3', '#d0e4f5': '#cfe2f3', '#c8dde8': '#cfe2f3', '#b8d4e8': '#cfe2f3', '#c8daf0': '#cfe2f3', '#8ab8da': '#4a90c4', '#5a7a99': '#4a90c4', '#4a7aab': '#4a90c4', '#3a7db8': '#4a90c4', '#3a6a99': '#4a90c4', '#2c6faa': '#4a90c4', '#2d6fa8': '#4a90c4', '#2a6099': '#4a90c4', '#1a5a8a': '#4a90c4', '#1a4f7a': '#4a90c4', '#1a4a7a': '#4a90c4', '#1e4f7a': '#4a90c4', '#1a3a5c': '#4a90c4', '#006395': '#4a90c4',
    
    # Data/sync (green)
    '#edf8f1': '#d0f0c0', '#edf7ee': '#d0f0c0', '#edf7f0': '#d0f0c0', '#e8f0e8': '#d0f0c0', '#e8f5eb': '#d0f0c0', '#f0fff0': '#d0f0c0', '#f5faf6': '#d0f0c0', '#aaddbb': '#d4edda', '#8acc98': '#d4edda', '#6ab87a': '#3d9e5a', '#4a9e5a': '#3d9e5a', '#4a7a5a': '#3d9e5a', '#2d9e44': '#3d9e5a', '#2d7a44': '#3d9e5a', '#2d7a3a': '#3d9e5a', '#2d7040': '#3d9e5a', '#266040': '#3d9e5a', '#1e5e35': '#3d9e5a', '#1a6e2e': '#3d9e5a', '#1a6a3a': '#3d9e5a', '#1a5c35': '#3d9e5a', '#1a5c30': '#3d9e5a', '#1a4a2a': '#3d9e5a',
    
    # Routing (orange)
    '#fff8f5': '#fdebd0', '#fff8f0': '#fdebd0', '#fff7ee': '#fdebd0', '#fdf6ee': '#fdebd0', '#fff3cd': '#fdebd0', '#ffcc00': '#c87b2a', '#b8860b': '#c87b2a', '#a05f1a': '#c87b2a', '#8a4000': '#c87b2a', '#7a5000': '#c87b2a', '#7a4a10': '#c87b2a', '#7a4010': '#c87b2a', '#7a3f00': '#c87b2a', '#7a3e00': '#c87b2a', '#5a3a10': '#c87b2a',
    
    # Error (red-pink)
    '#fff5f5': '#f9d6d5', '#fff0f0': '#f9d6d5', '#fdf5f5': '#f9d6d5', '#fdf0f0': '#f9d6d5', '#fde9e8': '#f9d6d5', '#e0d0d0': '#f9d6d5', '#cc4444': '#c44', '#900': '#c44', '#8b1a1a': '#c44', '#7a0e1a': '#c44', '#7a0c1c': '#c44', '#7a0000': '#c44',
    
    # MIT red accent
    'rgba(163,31,52,0.06)': '#f9d6d5',
    
    # Neutral
    '#fafafa': '#f7f7f7', '#f9f9f9': '#f7f7f7', '#f5f5f5': '#f7f7f7', '#f4f4f4': '#f7f7f7', '#f2f2f2': '#f7f7f7', '#f0f0f0': '#f7f7f7', '#eee': '#f7f7f7', '#d0d0d0': '#bbb', '#ccc': '#bbb', '#aaa': '#bbb', '#888': '#999', '#777': '#999', '#666': '#555', '#444': '#333', 'rgba(255,255,255,0.85)': '#fff', 'rgba(255,255,255,0.90)': '#fff', 'white': '#fff', '#ffffff': '#fff', '#bbbbbb': '#bbb', '#555555': '#555', '#333333': '#333', '#999999': '#999'
}

DEFS_XML = """
  <defs xmlns="http://www.w3.org/2000/svg">
    <marker id="arrow" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
      <path d="M0,0 L8,3 L0,6 Z" fill="#555"/>
    </marker>
    <marker id="arrow-red" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
      <path d="M0,0 L8,3 L0,6 Z" fill="#a31f34"/>
    </marker>
    <marker id="arrow-green" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
      <path d="M0,0 L8,3 L0,6 Z" fill="#3d9e5a"/>
    </marker>
  </defs>
"""

def map_color(color):
    if not color:
        return color
    color = color.lower()
    return COLOR_MAP.get(color, color)

def process_svg(filepath):
    try:
        parser = ET.XMLParser(remove_blank_text=False)
        tree = ET.parse(filepath, parser)
        root = tree.getroot()
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return False

    changed = False

    # 1. Enforce viewBox
    if root.attrib.get('viewBox') != '0 0 680 460':
        root.attrib['viewBox'] = '0 0 680 460'
        changed = True

    # 2. Enforce font-family
    if 'Helvetica Neue' not in root.attrib.get('font-family', ''):
        root.attrib['font-family'] = 'Helvetica Neue, Helvetica, Arial, sans-serif'
        changed = True

    # Get namespace
    nsmap = root.nsmap
    ns = nsmap.get(None, "http://www.w3.org/2000/svg")
    ns_prefix = f"{{{ns}}}"

    # 3. Enforce background rect
    has_bg = False
    for child in root:
        if isinstance(child.tag, str) and child.tag.endswith('rect'):
            w = child.attrib.get('width')
            h = child.attrib.get('height')
            if w == '680' and h in ('460', '470', '480'):
                child.attrib['height'] = '460'
                child.attrib['fill'] = '#fff'
                child.attrib['rx'] = '4'
                has_bg = True
                break
    if not has_bg:
        bg_rect = ET.Element(f"{ns_prefix}rect", {'width': '680', 'height': '460', 'fill': '#fff', 'rx': '4'})
        root.insert(0, bg_rect)
        changed = True

    # 4. Inject standard defs if missing
    has_defs = False
    for child in root:
        if isinstance(child.tag, str) and child.tag.endswith('defs'):
            has_defs = True
            break
    if not has_defs:
        defs_elem = ET.fromstring(DEFS_XML)
        # remove namespace prefix from defs_elem if needed to match root
        root.insert(1, defs_elem)
        changed = True

    # 5. Process all elements
    for elem in root.iter():
        if not isinstance(elem.tag, str):
            continue
            
        # Map colors
        for attr in ['fill', 'stroke']:
            if attr in elem.attrib:
                old_val = elem.attrib[attr]
                if not old_val.startswith('url'):
                    new_val = map_color(old_val)
                    if new_val != old_val:
                        elem.attrib[attr] = new_val
                        changed = True

        # Standardize rx
        if elem.tag.endswith('rect'):
            rx = elem.attrib.get('rx')
            if rx and rx not in ['2', '4', '5', '6', '12']:
                try:
                    rx_val = float(rx)
                    if rx_val < 3:
                        new_rx = '2'
                    elif rx_val < 4.5:
                        new_rx = '4'
                    elif rx_val < 5.5:
                        new_rx = '5'
                    elif rx_val < 9:
                        new_rx = '6'
                    else:
                        new_rx = '12'
                    if new_rx != rx:
                        elem.attrib['rx'] = new_rx
                        changed = True
                except ValueError:
                    pass

        # Scale down fonts
        if elem.tag.endswith('text') or elem.tag.endswith('tspan'):
            fs = elem.attrib.get('font-size')
            if fs:
                try:
                    fs_val = float(fs)
                    text_content = elem.text or ''
                    is_special = '·' in text_content or '...' in text_content or '×' in text_content
                    is_title = elem.attrib.get('y') in ['28', '30', '32'] and elem.attrib.get('x') == '340'
                    
                    if is_title:
                        if fs_val != 13:
                            elem.attrib['font-size'] = '13'
                            changed = True
                    elif not is_special and fs_val > 12:
                        elem.attrib['font-size'] = '12'
                        changed = True
                    elif not is_special and fs_val not in [7.5, 8, 8.5, 9, 9.5, 10, 11, 12]:
                        allowed = [7.5, 8, 8.5, 9, 9.5, 10, 11, 12]
                        closest = min(allowed, key=lambda x: abs(x - fs_val))
                        new_fs = str(int(closest)) if closest.is_integer() else str(closest)
                        if new_fs != fs:
                            elem.attrib['font-size'] = new_fs
                            changed = True
                except ValueError:
                    pass

    if changed:
        ET.cleanup_namespaces(root)
        xml_str = ET.tostring(root, encoding='utf-8', xml_declaration=True, pretty_print=False).decode('utf-8')
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(xml_str)
        return True
    return False

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        target_dir = sys.argv[1]
        svg_files = glob.glob(os.path.join(target_dir, '**/*.svg'), recursive=True)
        count = 0
        for f in svg_files:
            if process_svg(f):
                count += 1
        print(f"Processed {count} files in {target_dir}")
    else:
        print("Please provide a target directory.")

import os
import glob
from lxml import etree as ET

def polish_svg(filepath):
    try:
        parser = ET.XMLParser(remove_blank_text=False)
        tree = ET.parse(filepath, parser)
        root = tree.getroot()
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return False

    changed = False
    
    # Extract all rects
    rects = []
    for elem in root.iter():
        if isinstance(elem.tag, str) and elem.tag.endswith('rect'):
            try:
                x = float(elem.attrib.get('x', 0))
                y = float(elem.attrib.get('y', 0))
                w = float(elem.attrib.get('width', 0))
                h = float(elem.attrib.get('height', 0))
                if w > 0 and h > 0:
                    rects.append({'elem': elem, 'x': x, 'y': y, 'w': w, 'h': h, 'cx': x + w/2, 'cy': y + h/2})
            except ValueError:
                pass

    # Snap text to center
    for elem in root.iter():
        if isinstance(elem.tag, str) and elem.tag.endswith('text'):
            try:
                tx = float(elem.attrib.get('x', 0))
                ty = float(elem.attrib.get('y', 0))
                ta = elem.attrib.get('text-anchor', '')
                
                # Find if text is inside a rect
                for r in rects:
                    # If it's a background rect or very large, skip
                    if r['w'] >= 600 or r['h'] >= 400:
                        continue
                    
                    # Check if text is roughly centered horizontally in this rect
                    if abs(tx - r['cx']) < 15 and r['x'] <= tx <= r['x'] + r['w'] and r['y'] <= ty <= r['y'] + r['h'] + 10:
                        # If text-anchor is middle, snap x to exactly cx
                        if ta == 'middle' and tx != r['cx']:
                            elem.attrib['x'] = str(r['cx']).rstrip('0').rstrip('.') if '.' in str(r['cx']) else str(r['cx'])
                            changed = True
                        
                        # We won't auto-snap Y because there might be multiple lines of text
            except ValueError:
                pass

    # Snap orthogonal lines
    for elem in root.iter():
        if isinstance(elem.tag, str) and elem.tag.endswith('line'):
            try:
                x1 = float(elem.attrib.get('x1', 0))
                y1 = float(elem.attrib.get('y1', 0))
                x2 = float(elem.attrib.get('x2', 0))
                y2 = float(elem.attrib.get('y2', 0))
                
                # Check if it's almost horizontal
                if 0 < abs(y1 - y2) < 3:
                    avg_y = (y1 + y2) / 2
                    elem.attrib['y1'] = str(avg_y).rstrip('0').rstrip('.') if '.' in str(avg_y) else str(avg_y)
                    elem.attrib['y2'] = elem.attrib['y1']
                    changed = True
                # Check if it's almost vertical
                elif 0 < abs(x1 - x2) < 3:
                    avg_x = (x1 + x2) / 2
                    elem.attrib['x1'] = str(avg_x).rstrip('0').rstrip('.') if '.' in str(avg_x) else str(avg_x)
                    elem.attrib['x2'] = elem.attrib['x1']
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
            if polish_svg(f):
                count += 1
        print(f"Polished {count} files in {target_dir}")
    else:
        print("Please provide a target directory.")
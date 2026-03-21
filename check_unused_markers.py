import os
import glob
import xml.etree.ElementTree as ET

svg_files = glob.glob('book/quarto/contents/vol2/**/images/svg/*.svg', recursive=True)

for fpath in svg_files:
    if "/_" in fpath or fpath.startswith('_'): continue
    try:
        tree = ET.parse(fpath)
        root = tree.getroot()
        ns = {'svg': 'http://www.w3.org/2000/svg'}
        
        defined_markers = set()
        for marker in root.findall('.//svg:marker', ns):
            if 'id' in marker.attrib:
                defined_markers.add(marker.attrib['id'])
                
        if not defined_markers:
            continue
            
        used_markers = set()
        for elem in root.findall('.//svg:line', ns) + root.findall('.//svg:path', ns):
            for attr in ['marker-start', 'marker-mid', 'marker-end']:
                if attr in elem.attrib:
                    val = elem.attrib[attr]
                    if val.startswith('url(#'):
                        marker_id = val[5:-1]
                        used_markers.add(marker_id)
        
        unused = defined_markers - used_markers
        if unused:
            print(f"File {fpath}: defines unused markers {unused}")

    except Exception as e:
        print(f"Error reading {fpath}: {e}")

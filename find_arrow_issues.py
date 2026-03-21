import os
import glob
import xml.etree.ElementTree as ET

svg_files = glob.glob('book/quarto/contents/vol2/**/*.svg', recursive=True)

for fpath in svg_files:
    if "/_" in fpath or fpath.startswith('_'): continue
    try:
        tree = ET.parse(fpath)
        root = tree.getroot()
        ns = {'svg': 'http://www.w3.org/2000/svg'}
        
        # Check if markers are defined
        defined_markers = set()
        for marker in root.findall('.//svg:marker', ns):
            if 'id' in marker.attrib:
                defined_markers.add(marker.attrib['id'])
        
        # Check all lines and paths
        for elem in root.findall('.//svg:line', ns) + root.findall('.//svg:path', ns):
            for attr in ['marker-start', 'marker-mid', 'marker-end']:
                if attr in elem.attrib:
                    val = elem.attrib[attr]
                    if val.startswith('url(#'):
                        marker_id = val[5:-1]
                        if marker_id not in defined_markers:
                            print(f"File {fpath}: references undefined marker {marker_id}")

    except Exception as e:
        print(f"Error reading {fpath}: {e}")

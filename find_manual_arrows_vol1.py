import glob
import xml.etree.ElementTree as ET

svg_files = glob.glob('book/quarto/contents/vol1/**/*.svg', recursive=True)

for fpath in svg_files:
    if "/_" in fpath or fpath.startswith('_'): continue
    try:
        tree = ET.parse(fpath)
        root = tree.getroot()
        ns = {'svg': 'http://www.w3.org/2000/svg'}
        
        lines = root.findall('.//svg:line', ns)
        paths = root.findall('.//svg:path', ns)
        polygons = root.findall('.//svg:polygon', ns)
        
        if (lines or paths) and polygons:
            print(f"{fpath} has {len(lines) + len(paths)} lines/paths and {len(polygons)} polygons")
    except Exception as e:
        pass

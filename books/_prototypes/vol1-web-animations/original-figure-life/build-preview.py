#!/usr/bin/env python3
"""Build a local chapter preview; never modify the manuscript or published HTML."""
import argparse
import json
from pathlib import Path
import shutil

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[4]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('chapter_html', type=Path)
args = parser.parse_args()
chapter = args.chapter_html.resolve()
site = chapter.parents[2]
relative = chapter.relative_to(site)
output = ROOT / 'books/_build/original-figure-life'
shutil.copytree(site, output, dirs_exist_ok=True)
soup = BeautifulSoup(chapter.read_text(), 'html.parser')
outer = soup.find(id='fig-cnn')
assert outer is not None
figure = outer.find('figure')
image = figure.find('img')
svg_path = (chapter.parent / image['src']).resolve()
assert svg_path.suffix == '.svg' and svg_path.is_relative_to(site)
movement = outer['data-mlsys-movement-ms']
hold = outer['data-mlsys-reading-pause-ms']
assert 0 < int(movement) <= 60000 and 0 <= int(hold) <= 60000
for key in list(outer.attrs):
    if key.startswith(('data-mlsys-', 'data-explore')):
        del outer[key]
outer['data-original-convolution'] = ''
outer['data-movement-ms'] = movement
outer['data-reading-pause-ms'] = hold
template = soup.new_tag('template', attrs={'data-original-art': ''})
template.append(BeautifulSoup(svg_path.read_text(), 'xml').find('svg'))
outer.append(template)
example_path = Path(__file__).with_name('example-values.json')
example_data = json.loads(example_path.read_text())
assert soup.find(id=example_data['source_anchor']), 'Worked-example source anchor missing from chapter'
example = soup.new_tag('script', type='application/json', attrs={'data-example-values': ''})
example.string = json.dumps(example_data).replace('</', '<\\/')
outer.append(example)
css = soup.new_tag('link', rel='stylesheet', href='/_prototypes/vol1-web-animations/original-figure-life/original-figure.css')
soup.head.append(css)
script = soup.new_tag('script', src='/_prototypes/vol1-web-animations/original-figure-life/original-figure.js', defer=True)
soup.head.append(script)
destination = output / relative
destination.write_text(str(soup))
print(destination)

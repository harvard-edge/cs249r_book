#!/usr/bin/env python3
"""Local full-chapter review only; production enrollment is handled by the extension."""
import json
from pathlib import Path
import shutil

from bs4 import BeautifulSoup
import yaml

ROOT = Path(__file__).resolve().parents[4]
record = json.loads(Path('/tmp/figures-animation-art-current/index.json').read_text())['vol3:fig-vol3-context-zones']
chapter = Path(record['html'])
site = chapter.parents[2]
output = ROOT / 'books/_build/original-context-invalidation'
shutil.copytree(site, output, dirs_exist_ok=True)
soup = BeautifulSoup(chapter.read_text(), 'html.parser')
outer = soup.find(id='fig-vol3-context-zones')
assert outer
for key in list(outer.attrs):
    if key.startswith(('data-mlsys-', 'data-explore')):
        del outer[key]
timing = yaml.safe_load((ROOT / 'books/config/shared/html/figure-explore.yml').read_text())['figure-explore']
outer['data-mlsys-movement-ms'] = str(timing['movement-ms'])
outer['data-mlsys-reading-pause-ms'] = str(timing['reading-pause-ms'])
figure = outer.find('figure')
image = figure.find('img')
assert image and 'context_staging_layout.svg' in image['src']
image['data-context-static'] = ''
template = soup.new_tag('template', attrs={'data-context-art': ''})
template.append(BeautifulSoup(Path(record['asset']).read_text(), 'xml').find('svg'))
outer.append(template)
for filename in ['original-player.css', 'original-context-invalidation.css']:
    soup.head.append(soup.new_tag('link', rel='stylesheet', href=f'/_extensions/mlsysbook-ext/figure-explore/{filename}'))
style = soup.new_tag('style')
style.string = '#fig-vol3-context-zones[data-context-ready] img[data-context-static]{display:none}@media print{#fig-vol3-context-zones[data-context-ready] img[data-context-static]{display:block!important}}'
soup.head.append(style)
for filename in ['original-player.js', 'original-context-invalidation.js']:
    soup.head.append(soup.new_tag('script', src=f'/_extensions/mlsysbook-ext/figure-explore/{filename}', defer=True))
script = soup.new_tag('script', defer=True)
script.string = """document.addEventListener('DOMContentLoaded',async()=>{
const outer=document.getElementById('fig-vol3-context-zones'),figure=outer.querySelector('figure'),root=document.createElement('div');
figure.prepend(root);
try{window.contextPlayer=await window.MLSysFigureLessons['original-context-invalidation'](root,{outer,figure,verifiedArtwork:outer.querySelector('template').content.querySelector('svg')});outer.dataset.contextReady='';}
catch(error){root.remove();console.error(error);}
});"""
soup.head.append(script)
destination = output / chapter.relative_to(site)
destination.write_text(str(soup))
print(destination)

#!/usr/bin/env python3
"""Build a QA-only full-chapter fixture without copying its rendered site."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import re
import struct
from urllib.parse import quote

from bs4 import BeautifulSoup
import yaml

ROOT = Path(__file__).resolve().parents[3]
BOOKS = ROOT / 'books'
EXT = BOOKS / '_extensions/mlsysbook-ext/figure-explore'


def javascript(value):
    return json.dumps(value, ensure_ascii=True).replace('<', '\\u003c')


def build(args):
    entry_path = args.entry.resolve()
    entry = json.loads(entry_path.read_text())
    key, implementation = entry['key'], entry['implementation']
    record = json.loads(args.art_index.read_text())[key]
    chapter = Path(record['html']).resolve()
    site = chapter.parents[2]
    asset = Path(record['asset']).resolve()
    data = asset.read_bytes()
    kind = 'png' if data.startswith(b'\x89PNG\r\n\x1a\n') else 'svg'
    if kind == 'png':
        width, height = struct.unpack('>II', data[16:24])
        artwork = {'kind':kind, 'width':width, 'height':height,
                   'url':'data:image/png;base64,' + base64.b64encode(data).decode()}
    else:
        artwork = {'kind':kind, 'text':data.decode('utf-8')}
    link_id = hashlib.sha256(str(site).encode()).hexdigest()[:16]
    link = BOOKS / '_build/fixture-sources' / link_id
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        if link.resolve() != site:
            raise ValueError(f'Fixture source symlink has unexpected target: {link}')
    elif link.exists():
        raise ValueError(f'Fixture source path already exists and is not a symlink: {link}')
    else:
        link.symlink_to(site, target_is_directory=True)
    output = args.output or BOOKS / '_build/original-rollout-fixtures' / entry_path.parent.name
    output = output.resolve()
    if output != BOOKS / '_build' and BOOKS / '_build' not in output.parents:
        raise ValueError('Fixture output must remain inside ignored books/_build.')
    output.mkdir(parents=True, exist_ok=True)
    soup = BeautifulSoup(chapter.read_text(), 'html.parser')
    for element in soup.select('[data-mlsys-explore], [data-explore]'):
        for name in ('data-mlsys-explore', 'data-explore'):
            if name in element.attrs:
                element['data-fixture-disabled-' + name[5:]] = element.attrs.pop(name)
    figure_id = key.split(':', 1)[1]
    outer = soup.find(id=figure_id)
    if outer is None:
        raise ValueError(f'Missing original figure {figure_id} in {chapter}')
    figure = outer if outer.name == 'figure' else outer.find('figure')
    if figure is None or figure.find('figcaption', recursive=False) is None:
        raise ValueError('Fixture requires the original numbered figure and caption.')
    timing = yaml.safe_load((ROOT / 'books/config/shared/html/figure-explore.yml').read_text())['figure-explore']
    outer['data-mlsys-movement-ms'] = str(timing['movement-ms'])
    outer['data-mlsys-reading-pause-ms'] = str(timing['reading-pause-ms'])
    outer['data-fixture-key'] = key
    outer['data-fixture-qa-only'] = 'true'
    outer['data-mlsys-explore-key'] = key
    for base in soup.find_all('base'):
        base.decompose()
    relative = chapter.parent.relative_to(site).as_posix()
    base_url = args.base_url.rstrip('/')
    soup.head.insert(0, soup.new_tag('base', href=f'{base_url}/_build/fixture-sources/{link_id}/{quote(relative)}/'))
    for filename in ['figure-views.css', 'original-player.css', implementation['stylesheet']]:
        if not (EXT / filename).is_file():
            raise FileNotFoundError(EXT / filename)
        soup.head.append(soup.new_tag('link', rel='stylesheet', href=f'/_extensions/mlsysbook-ext/figure-explore/{quote(filename)}'))
    data_scripts = list(implementation.get('data_scripts', []))
    if implementation.get('data_script'):
        data_scripts = [implementation['data_script'], *data_scripts]
    data_scripts += args.data_script
    if implementation.get('bundle'):
        data_scripts.insert(0, implementation['bundle'])
    data_scripts = list(dict.fromkeys(data_scripts))
    for filename in ['figure-views.js', 'original-player.js', *data_scripts, implementation['script']]:
        if not (EXT / filename).is_file():
            raise FileNotFoundError(EXT / filename)
        soup.head.append(soup.new_tag('script', src=f'/_extensions/mlsysbook-ext/figure-explore/{quote(filename)}', defer=True))
    # Reuse the frozen loader's exact fragment-isolation algorithm without exposing
    # a new production API or executing the production mount/freshness pipeline.
    loader = (EXT / 'figure-explore.js').read_text()
    namespace = re.search(r'  const namespaceArtwork =[\s\S]*?(?=  const mount =)', loader)
    if namespace is None:
        raise ValueError('Current SVG namespace helper was not found; update fixture builder.')
    payload = {'id':figure_id, 'name':implementation['name'], 'art':artwork}
    script = soup.new_tag('script')
    script.string = '''document.addEventListener('DOMContentLoaded',async()=>{
'use strict';
let svgSerial=0;
''' + namespace.group() + '\nconst config=' + javascript(payload) + ''';
const outer=document.getElementById(config.id),figure=outer.matches('figure')?outer:outer.querySelector('figure');
const caption=figure.querySelector(':scope > figcaption'),original=[...figure.childNodes];
const diagram=document.createElement('div'),explore=document.createElement('div');
diagram.id=config.id+'-fixture-diagram';explore.id=config.id+'-fixture-explore';explore.hidden=true;
window.fixtureRoot=explore;
try{
 let art;
 if(config.art.kind==='png'){
  const image=new Image();image.src=config.art.url;await image.decode();
  if(image.naturalWidth!==config.art.width||image.naturalHeight!==config.art.height)throw Error('Fixture PNG dimensions disagree');
  const ns='http://www.w3.org/2000/svg';art=document.createElementNS(ns,'svg');
  art.setAttribute('viewBox',`0 0 ${config.art.width} ${config.art.height}`);
  art.setAttribute('width',config.art.width);art.setAttribute('height',config.art.height);art.setAttribute('preserveAspectRatio','xMidYMid meet');
  art.dataset.mlsysArtworkKind='png';art.dataset.mlsysArtworkWidth=config.art.width;art.dataset.mlsysArtworkHeight=config.art.height;
  const raster=document.createElementNS(ns,'image');for(const [name,value]of Object.entries({x:0,y:0,width:config.art.width,height:config.art.height,href:config.art.url}))raster.setAttribute(name,value);art.append(raster);
 }else{
  const parsed=new DOMParser().parseFromString(config.art.text,'image/svg+xml');
  if(parsed.querySelector('parsererror')||parsed.documentElement.localName!=='svg')throw Error('Invalid fixture SVG');
  art=document.importNode(parsed.documentElement,true);
 }
 namespaceArtwork(art,config.id+'-fixture');
 const factory=window.MLSysFigureLessons?.[config.name];if(typeof factory!=='function')throw Error('Fixture factory unavailable: '+config.name);
 window.fixturePlayer=await factory(explore,{outer,figure,staticNodes:original.filter(n=>n!==caption),verifiedArtwork:art});
 if(typeof window.fixturePlayer?.pause!=='function')throw Error('Fixture factory requires pause()');
 for(const node of original)if(node!==caption)diagram.append(node);
 figure.insertBefore(diagram,caption);figure.insertBefore(explore,caption);
 figure.addEventListener('mlsys-view-change',event=>{if(event.detail.view!=='explore')window.fixturePlayer.pause();});
 window.MLSysFigureViews.enhance({figure,diagram,explore,initialView:'explore'});
 window.fixturePlayer.pause();outer.dataset.fixtureState='ready';
}catch(error){window.fixturePlayer?.destroy?.();figure.replaceChildren(...original);figure.classList.remove('mlsys-figure-views');delete figure.dataset.figureViewsReady;delete figure.dataset.figureView;outer.dataset.fixtureState='failed';window.fixtureError=String(error);console.error('[QA-only original-art fixture]',error);}
});'''
    soup.head.append(script)
    destination = output / 'index.html'
    destination.write_text(str(soup))
    result = {'path':str(destination), 'url':base_url + '/' + destination.relative_to(BOOKS).as_posix(),
              'key':key, 'qa_only':True, 'source_chapter':str(chapter),
              'source_art_sha256':hashlib.sha256(data).hexdigest()}
    (output / 'fixture.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--entry', type=Path, required=True)
    parser.add_argument('--art-index', type=Path, default=Path('/tmp/figures-animation-art-current/index.json'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--base-url', default='http://127.0.0.1:8882')
    parser.add_argument('--data-script', action='append', default=[], help='Extension-relative data JS loaded before the factory (repeatable).')
    args = parser.parse_args()
    print(json.dumps(build(args)))


if __name__ == '__main__':
    main()

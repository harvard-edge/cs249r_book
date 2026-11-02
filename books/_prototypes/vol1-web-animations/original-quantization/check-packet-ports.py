"""Inspect every original-quantization moving edge without changing lesson timing.

The fixture mounts in the full existing chapter. It captures the real transition's
registered frame callbacks after pausing, then samples those same callbacks at
declared progress fractions. Production packaging is verified separately.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
from playwright.sync_api import sync_playwright
import yaml

REPO=Path(__file__).resolve().parents[4]
ASSETS=REPO/'books/_extensions/mlsysbook-ext/figure-explore'
parser=argparse.ArgumentParser()
parser.add_argument('--engine',choices=['chromium','webkit'],default='chromium')
parser.add_argument('--output',default='/tmp/quantization-packet-ports')
args=parser.parse_args()
out=Path(args.output);out.mkdir(exist_ok=True,parents=True)
entry=json.loads(Path('/tmp/figures-animation-art-current/index.json').read_text())['vol1:fig-weight-activations-quantization']
timing=yaml.safe_load((REPO/'books/config/shared/html/figure-explore.yml').read_text())['figure-explore']
js=(ASSETS/'original-quantization.js').read_text()
baseline=subprocess.check_output(['git','show','HEAD:books/_extensions/mlsysbook-ext/figure-explore/original-quantization.js'],cwd=REPO,text=True)
semantic=lambda text:text.split('  const steps=[')[1].split('  // Dock outside')[0].split('  const routes=')[0]
assert semantic(js)==semantic(baseline), 'steps/equations/render semantics changed'
for name in ['original-quantization.data.js','original-quantization.calc.py']:
 assert (ASSETS/name).read_bytes()==subprocess.check_output(['git','show','HEAD:books/_extensions/mlsysbook-ext/figure-explore/'+name],cwd=REPO), 'numerical source changed'
spec=importlib.util.spec_from_file_location('calcs',ASSETS/'original-quantization.calc.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
report={'engine':args.engine,'hashes':{name:hashlib.sha256((ASSETS/name).read_bytes()).hexdigest() for name in ['original-quantization.js','original-quantization.css','original-quantization.data.js','original-quantization.calc.py']},'semantic_source_unchanged':True,'profiles':[]}
with sync_playwright() as p:
 browser=getattr(p,args.engine).launch()
 for width,theme in [(1440,'light'),(1440,'dark'),(390,'light'),(390,'dark')]:
  page=browser.new_page(viewport={'width':width,'height':1800 if width>700 else 1100},color_scheme=theme)
  page.goto(Path(entry['html']).as_uri(),wait_until='load')
  page.evaluate("""theme=>{document.body.classList.toggle('quarto-dark',theme==='dark');document.body.classList.toggle('quarto-light',theme==='light');document.documentElement.setAttribute('data-bs-theme',theme);document.querySelectorAll('link.quarto-color-scheme').forEach(l=>{l.disabled=theme==='dark'?!l.classList.contains('quarto-color-alternate'):l.classList.contains('quarto-color-alternate')});}""",theme)
  for name in ['original-player.css','original-quantization.css']:page.add_style_tag(path=str(ASSETS/name))
  for name in ['original-player.js','original-quantization.data.js','original-quantization.js']:page.add_script_tag(path=str(ASSETS/name))
  assert page.evaluate('MLSysOriginalQuantizationData')==module.calculate()
  page.evaluate("""async config=>{
   const outer=document.getElementById('fig-weight-activations-quantization');const cap=outer.querySelector('figcaption')?.cloneNode(true);outer.replaceChildren();const root=document.createElement('div');outer.append(root);if(cap)outer.append(cap);outer.dataset.mlsysMovementMs=String(config.movement);outer.dataset.mlsysReadingPauseMs=String(config.reading);
   const create=MLSysOriginalPlayer.create;window.MLSysOriginalPlayer={...MLSysOriginalPlayer,create(options){const transition=options.transition;options.transition=context=>{window.packetFrames=[];return transition({...context,onFrame(fn){packetFrames.push(fn);context.onFrame(fn);}});};return create(options);}};
   const art=document.importNode(new DOMParser().parseFromString(config.svg,'image/svg+xml').documentElement,true);window.originalChildren=Array.from(art.children).map(n=>n.outerHTML);window.quantRoot=root;window.quantPlayer=await MLSysFigureLessons['original-quantization'](root,{outer,verifiedArtwork:document.importNode(art,true)});outer.scrollIntoView({block:'center'});
  }""",{'svg':Path(entry['asset']).read_text(),'movement':timing['movement-ms'],'reading':timing['reading-pause-ms']})
  page.wait_for_timeout(100)
  assert page.evaluate("JSON.stringify(originalChildren)===JSON.stringify(Array.from(quantRoot.querySelector('svg').children).filter(n=>!n.classList.contains('oq-overlay')).map(n=>n.outerHTML))"), 'canonical SVG descendants changed'
  samples=[]
  for target in range(1,9):
   page.evaluate('(t)=>{quantPlayer.go(t-1);quantPlayer.next();quantPlayer.pause();}',target)
   assert page.locator('.oq-token').count()==(2 if target==3 else 1)
   for progress in [.02,.25,.5,.75,.98]:
    page.evaluate('(v)=>packetFrames.forEach(fn=>fn(v))',progress)
    collisions=page.evaluate("""()=>{
     const overlap=(a,b)=>a.left<b.right-.1&&a.right>b.left+.1&&a.top<b.bottom-.1&&a.bottom>b.top+.1;
     const boxes=Array.from(quantRoot.querySelectorAll('[data-oq-box]>rect'));
     const glyphs=Array.from(quantRoot.querySelectorAll('svg use'));
     return Array.from(quantRoot.querySelectorAll('.oq-packet,.oq-token-badge')).flatMap(packet=>[...boxes,...glyphs].filter(other=>overlap(packet.getBoundingClientRect(),other.getBoundingClientRect())).map(other=>({packet:packet.className.baseVal,other:other.parentNode.getAttribute('data-oq-box')||'original glyph'})));
    }""")
    assert not collisions,(width,theme,target,progress,collisions)
    before=page.locator('.oq-token').first.get_attribute('transform');page.wait_for_timeout(25);assert page.locator('.oq-token').first.get_attribute('transform')==before
    assert page.evaluate('quantRoot.scrollWidth<=quantRoot.clientWidth+1')
    if progress in [.02,.5,.98]:
     page.locator('.oq-canvas').screenshot(path=str(out/f'{args.engine}-{width}-{theme}-edge-{target}-p-{progress}.png'))
    samples.append({'edge':target,'progress':progress,'collision_count':len(collisions),'paused':True})
   page.evaluate('(t)=>quantPlayer.go(t)',target)
  page.locator('#fig-weight-activations-quantization').screenshot(path=str(out/f'{args.engine}-{width}-{theme}-final-chapter.png'))
  report['profiles'].append({'width':width,'theme':theme,'samples':samples})
  print(args.engine,width,theme,'PASS: every moving edge, five progress samples, no packet/badge overlap with box or original glyph; pause frozen')
  page.close()
 browser.close()
(out/f'{args.engine}-report.json').write_text(json.dumps(report,indent=2)+'\n')

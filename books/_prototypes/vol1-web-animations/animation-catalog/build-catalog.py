"""Validate reviewed proposals against the inventory and render a local review catalog."""
from pathlib import Path
import json,re,subprocess,os,argparse
from datetime import date
from collections import Counter
from bs4 import BeautifulSoup
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--build-root',type=Path,help='Successful four-volume parallel HTML run (relative to repository root or absolute).')
parser.add_argument('--output-dir',type=Path,default=HERE,help='Optional QA output directory; defaults to the catalog directory.')
args=parser.parse_args()
EXT=ROOT/'books/_extensions/mlsysbook-ext/figure-explore'
registry=json.loads((EXT/'figure-registry.json').read_text())
lessons={}
# Dedicated implementations replace the reserved guided rows, as in the compiler.
for path in sorted((EXT/'lessons').glob('vol*-*.json'))+sorted((EXT/'lessons').glob('specialized-vol*.json')):
 for lesson in json.loads(path.read_text()): lessons[lesson['key']]=lesson

def successful_run(path):
 summary=path/'summary.json'
 if not summary.is_file(): return False
 jobs=json.loads(summary.read_text()).get('jobs',[])
 return all(any(j.get('volume')==f'vol{v}' and j.get('format')=='html' and j.get('ok') and j.get('chapter') is None for j in jobs) for v in range(1,5))
if args.build_root:
 build=args.build_root if args.build_root.is_absolute() else ROOT/args.build_root
 if not successful_run(build): parser.error('--build-root must name a successful complete four-volume HTML run')
else:
 candidates=sorted((ROOT/'books/_build/parallel').glob('*'),reverse=True)
 build=next((p for p in candidates if successful_run(p)),None)
 if build is None: parser.error('No successful four-volume HTML build found; provide --build-root')
output=args.output_dir.resolve();output.mkdir(parents=True,exist_ok=True)
files=['vol1-a-proposals.json','vol1-b-proposals.json','vol2-a-proposals.json','vol2-b-proposals.json','vol3-proposals.json','vol4-proposals.json']
inventory=json.loads((HERE/'inventory.json').read_text())
proposals=[r for name in files for r in json.loads((HERE/name).read_text())]
keys=[r['key'] for r in proposals];assert len(keys)==len(set(keys)), 'Duplicate proposals'
by_key={r['key']:r for r in proposals}
assert set(by_key)=={r['key'] for r in inventory},f"Coverage mismatch: missing {set(r['key'] for r in inventory)-set(by_key)}, extra {set(by_key)-set(r['key'] for r in inventory)}"
for r in proposals:
 for field in ['systems_goal','visual','fidelity_guard']:assert isinstance(r[field],str) and len(r[field])>15,(r['key'],field)
 assert r['treatment'] in ['animated','interactive','annotated','static']
 assert r['priority'] in ['pilot','next','later','keep']
 assert r['family'] in ['memory-movement','lifetime','scheduling','computation','parameter-curve','topology','annotation','static'], (r['key'],r['family'])
 assert len(r['storyboard'])==3 and all(isinstance(s,str) and len(s)>5 for s in r['storyboard'])
first_wave=['vol1:fig-tensor-memory-layout','vol1:fig-tiling-diagram','vol1:fig-activation-checkpointing','vol1:fig-dataloader-choke-point','vol2:fig-hierarchical-allreduce','vol2:fig-comm-compute-overlap','vol2:fig-storage-checkpoint-staging','vol2:fig-paged-attention','vol3:fig-vol3-pagedattention-mapping','vol3:fig-vol3-idempotency-flow','vol3:fig-vol3-intent-before-dispatch','vol3:fig-vol3-hierarchical-budget','vol4:fig-04-multirate-cadence-buffer','vol4:fig-09-ray-update-epochs','vol4:fig-enforcement-stopping-budget','vol4:fig-16-release-record']
assert set(first_wave)<=set(by_key),set(first_wave)-set(by_key)
rows=[]
chapter_cache={}
for original in inventory:
 row={**original,**by_key[original['key']]};row.pop('context',None)
 row['first_wave']=row['key'] in first_wave
 lesson=lessons.get(row['key'],{})
 registered=registry.get(row['key'])
 row['proposed_treatment']=row['treatment']
 row['proposal_review_basis']='Caption, alt description, and local source context; original proposal retained as planning evidence.'
 row['actual_treatment']=('guided-annotation' if registered['name']=='guided-figure' else 'specialized-model') if registered else ('static' if lesson.get('status')=='static' else 'source-conflict' if lesson.get('status')=='blocked' else 'unresolved')
 row['status']='registered' if registered else lesson.get('status','unresolved')
 row['actual_status']=row['status']
 row['review_basis']=lesson.get('review_notes','Previously reviewed specialized memory-layout / matrix-tiling implementation; canonical artwork retained.')
 row['actual_review_basis']=row['review_basis']
 row['artwork_reviewed']=lesson.get('artwork_reviewed',bool(registered))
 row['runtime_verification']='Full runtime sweep passed for 686 enrolled players. Refined artwork projection and capture passed for all 686 in four profiles (2,744 screens), with recorded infrastructure retries. All 16 specialized players passed a separate full runtime recheck. This is not a semantic certification.' if registered else 'Static entry; excluded from interactive runtime checks.'
 row['visual_verification']='Authors inspected all 670 guided figures in actual chapter desktop-light and narrow-dark captures, with representative alternate profiles. Specialized lessons received separate state, numerical, and visual audits. Final label and focus refinements receive targeted actual-theme rechecks; no blanket semantic certification is claimed.' if registered else 'Retained static or held for a documented source conflict.'
 row['actual_question']=lesson.get('question','')
 row['actual_scope']=lesson.get('scope','Specialized stateful mechanism lesson; its declared assumptions and limits appear inside the player.' if registered else '')
 row['actual_steps']=[{**step,'focus':[lesson.get('regions',{}).get(key,{}).get('label',key) for key in step.get('focus',[])]} for step in lesson.get('steps',[])]
 chapter=build/f"vol{row['volume']}-html/output"/Path(row['source']).relative_to('books').with_suffix('.html')
 row['chapter_url']=os.path.relpath(chapter,output)+'#'+row['figure_id'] if registered and chapter.is_file() else ''
 row['built_explore_present']=False
 if row['chapter_url']:
  if chapter not in chapter_cache: chapter_cache[chapter]=BeautifulSoup(chapter.read_text(),'html.parser')
  node=chapter_cache[chapter].find(id=row['figure_id'])
  row['built_explore_present']=bool(node and node.get('data-mlsys-explore'))

 row['metadata_issue']='Malformed source figure attributes: caption contains fig-alt=. Repair and re-review before implementation.' if 'fig-alt=' in row['caption'] else ''
 rows.append(row)
meta={'date':date.today().isoformat(),'source_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'scope':'Named body figures in the four canonical PDF chapter/appendix lists, plus Binder-generated index sources. Covers and unnumbered margin art are excluded.','count':len(rows),'counts_by_volume':dict(Counter(str(r['volume']) for r in rows)),'treatments':dict(Counter(r['treatment'] for r in rows)),'first_wave':first_wave,'actual_treatments':dict(Counter(r['actual_treatment'] for r in rows)),'actual_statuses':dict(Counter(r['actual_status'] for r in rows)),'build_root':str(build.relative_to(ROOT)),'review_limit':'Original-art/source inspection and peer feedback inform authored lessons. Guided annotations preserve the source image; they are not simulations. Runtime sweep and refined projection/capture passed for 686 enrolled players in four profiles (2,744 screens), with documented infrastructure retries; all 16 specialized lessons passed a separate runtime recheck. Authors inspected all guided figures in actual chapter desktop-light and narrow-dark captures plus representative alternate profiles. Final label and focus refinements receive targeted actual-theme rechecks. Automated passage is not a semantic certification.'}
(output/'catalog.json').write_text(json.dumps({'metadata':meta,'figures':rows},ensure_ascii=False,indent=2))
base=build/'vol1-html/output/index.html'
bs=BeautifulSoup(base.read_text(),'html.parser')
links=[]
for selector,ident in [('#quarto-bootstrap','catalog-light'),('link.quarto-color-alternate[href*="bootstrap-dark"]','catalog-dark')]:
 node=bs.select_one(selector);asset=(base.parent/node['href']).resolve();links.append(f'<link id="{ident}" rel="stylesheet" href="{os.path.relpath(asset,output)}"'+(' disabled' if ident=='catalog-dark' else '')+'>')
html=(HERE/'catalog-template.html').read_text().replace('<!-- THEME -->','\n'.join(links)).replace('/* CATALOG_DATA */',json.dumps({'metadata':meta,'figures':rows},ensure_ascii=False).replace('</','<\\/'))
(output/'index.html').write_text(html)
print(json.dumps({'coverage':len(rows),'volume':meta['counts_by_volume'],'proposed_treatments':meta['treatments'],'actual_treatments':meta['actual_treatments'],'actual_statuses':meta['actual_statuses'],'build_root':meta['build_root'],'first_wave':len(first_wave),'metadata_issues':sum(bool(r['metadata_issue']) for r in rows)},indent=2))

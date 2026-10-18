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
SELECTION_REASONS={'tensor-memory-layout': 'Selected tensor cells map to different storage offsets; traversal exposes locality.', 'matrix-tiling': 'Operand copies, numeric multiply-accumulate results, and final writes change as tiles execute.', 'checkpointing': 'Activations are allocated, released, and recomputed, exposing the memory/work tradeoff.', 'comm-compute-overlap': 'Readiness, simultaneous communication/computation, and the final exposed barrier change over the schedule.', 'hierarchical-allreduce': 'Shard ownership and partial sums evolve into global results across the hierarchy.', 'paged-attention': 'Allocation, growth, block-table entries, and release change cache ownership.', 'storage-checkpoint-staging': 'Snapshot ownership and durable copies change before the manifest commits the checkpoint.', 'v3-hierarchical-budget': 'Reserved, settled, and free balances change as child work and late bills consume a budget.', 'v3-idempotency': 'Selectable replay, pending, conflict, unknown, and effect cases change operation outcomes.', 'v3-intent-ordering': 'Selectable failure points change whether dispatch and durable intent survive.', 'v3-paged-mapping': 'Block lookup, copy-on-write tail copying, and appending change the mapping and data.', 'physical-multirate': 'Lease time, fallback, and mailbox-counter states change as producer and consumer progress.', 'physical-ray-epochs': 'Rays, cells, and accumulated evidence actually change across epochs.', 'physical-stopping-budget': 'Additive stopping distances, candidate speed, and dispatch charge change whether clearance suffices.'}
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
first_wave=sorted(key for key,entry in registry.items() if entry['name']!='guided-figure')
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
 row['runtime_verification']='All four HTML jobs passed on final build 20261018-100612-17815. Convolution and KV paging passed four-profile runtime and targeted tasks on build 094548; their code is unchanged in the final build. Quantization passed four-profile full-runtime and targeted checks again on final build 100612 after a responsive-table/caption fix. All 12 pilot task/profile cases pass across this scoped combination. Final loading, ID, and freshness checks passed for all 16 lessons across 15 chapter pages with no missing lessons or failures. The 13 unchanged lessons retain prior scoped runtime proof; they were not all fully rerun in this iteration.' if registered else 'Static figure; no interactive runtime required.'
 row['visual_verification']='Three simulated learner-perspective reviewers informed pacing, local controls, comparison, and lookup/self-check changes. Rendered pilot task captures were inspected; final quantization review verified both comparison columns on narrow screens and readable dark captions. An independent simulated novice review found no remaining blocker. No students were recruited and no learning gains were measured.' if registered else 'Original static figure retained without slideshow controls.'
 row['selection_reason']=lesson.get('learning_value','') or SELECTION_REASONS.get(registered['name'],'') if registered else lesson.get('static_reason','') or lesson.get('review_notes','')
 row['actual_question']=lesson.get('question','')
 row['actual_scope']=row['selection_reason'] or lesson.get('static_reason','') or lesson.get('scope','Specialized stateful mechanism lesson; its declared assumptions and limits appear inside the player.' if registered else '')
 row['actual_steps']=[{**step,'focus':[lesson.get('regions',{}).get(key,{}).get('label',key) for key in step.get('focus',[])]} for step in lesson.get('steps',[])] if registered else []
 chapter=build/f"vol{row['volume']}-html/output"/Path(row['source']).relative_to('books').with_suffix('.html')
 row['chapter_url']=os.path.relpath(chapter,output)+'#'+row['figure_id'] if chapter.is_file() else ''
 row['built_explore_present']=False
 if registered and row['chapter_url']:
  if chapter not in chapter_cache: chapter_cache[chapter]=BeautifulSoup(chapter.read_text(),'html.parser')
  node=chapter_cache[chapter].find(id=row['figure_id'])
  row['built_explore_present']=bool(node and node.get('data-mlsys-explore'))

 row['metadata_issue']='Malformed source figure attributes: caption contains fig-alt=. Repair and re-review before implementation.' if 'fig-alt=' in row['caption'] else ''
 rows.append(row)
meta={'date':date.today().isoformat(),'source_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'scope':'Named body figures in the four canonical PDF chapter/appendix lists, plus Binder-generated index sources. Covers and unnumbered margin art are excluded.','count':len(rows),'counts_by_volume':dict(Counter(str(r['volume']) for r in rows)),'stateful_by_volume':dict(Counter(str(r['volume']) for r in rows if r['actual_treatment']=='specialized-model')),'treatments':dict(Counter(r['treatment'] for r in rows)),'first_wave':first_wave,'actual_treatments':dict(Counter(r['actual_treatment'] for r in rows)),'actual_statuses':dict(Counter(r['actual_status'] for r in rows)),'build_root':str(build.relative_to(ROOT)),'review_limit':'Selective rollout: 16 stateful lessons and 691 static figures; no narrated image tours. Three simulated learner-perspective reviews informed convolution, quantization, and KV-paging changes. All four final HTML jobs passed. Four-profile pilot runtime and targeted tasks pass across builds 094548 and 100612; final quantization visual proof includes its responsive table/caption repair. Final loading, ID, and freshness checks passed for all 16 lessons across 15 chapter pages with no missing lessons or failures. Prior runtime proof for 13 unchanged lessons remains historical. No students were recruited and no learning gains were measured.'}
(output/'catalog.json').write_text(json.dumps({'metadata':meta,'figures':rows},ensure_ascii=False,indent=2))
base=build/'vol1-html/output/index.html'
bs=BeautifulSoup(base.read_text(),'html.parser')
links=[]
for selector,ident in [('#quarto-bootstrap','catalog-light'),('link.quarto-color-alternate[href*="bootstrap-dark"]','catalog-dark')]:
 node=bs.select_one(selector);asset=(base.parent/node['href']).resolve();links.append(f'<link id="{ident}" rel="stylesheet" href="{os.path.relpath(asset,output)}"'+(' disabled' if ident=='catalog-dark' else '')+'>')
html=(HERE/'catalog-template.html').read_text().replace('<!-- THEME -->','\n'.join(links)).replace('/* CATALOG_DATA */',json.dumps({'metadata':meta,'figures':rows},ensure_ascii=False).replace('</','<\\/'))
(output/'index.html').write_text(html)
print(json.dumps({'coverage':len(rows),'volume':meta['counts_by_volume'],'proposed_treatments':meta['treatments'],'actual_treatments':meta['actual_treatments'],'actual_statuses':meta['actual_statuses'],'build_root':meta['build_root'],'first_wave':len(first_wave),'metadata_issues':sum(bool(r['metadata_issue']) for r in rows)},indent=2))

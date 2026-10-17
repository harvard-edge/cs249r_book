"""Validate reviewed proposals against the inventory and render a local review catalog."""
from pathlib import Path
import json,re,subprocess,os
from collections import Counter
from bs4 import BeautifulSoup
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
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
for original in inventory:
 row={**original,**by_key[original['key']]};row.pop('context',None)
 row['first_wave']=row['key'] in first_wave
 row['review_basis']='Caption, alt description, and local source context; original artwork must be inspected before implementation.'
 row['status']='Local working pilot' if row['key'] in ['vol1:fig-cnn','vol1:fig-tensor-memory-layout'] else 'Proposal'
 row['metadata_issue']='Malformed source figure attributes: caption contains fig-alt=. Repair and re-review before implementation.' if 'fig-alt=' in row['caption'] else ''
 rows.append(row)
meta={'date':'2026-10-17','source_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'scope':'Named body figures in the four canonical PDF chapter/appendix lists, plus Binder-generated index sources. Covers and unnumbered margin art are excluded.','count':len(rows),'counts_by_volume':dict(Counter(str(r['volume']) for r in rows)),'treatments':dict(Counter(r['treatment'] for r in rows)),'first_wave':first_wave,'review_limit':'Planning proposals, not 707 implemented or visually approved animations. Six proposal reviewers read assigned figure metadata and local context. Rendered artwork verification remains an implementation gate.'}
(HERE/'catalog.json').write_text(json.dumps({'metadata':meta,'figures':rows},ensure_ascii=False,indent=2))
base=ROOT/'books/_build/html-vol1/chapters/06_nn_architectures/vol1/06_nn_architectures/06_nn_architectures.html'
bs=BeautifulSoup(base.read_text(),'html.parser')
links=[]
for selector,ident in [('#quarto-bootstrap','catalog-light'),('link.quarto-color-alternate[href*="bootstrap-dark"]','catalog-dark')]:
 node=bs.select_one(selector);asset=(base.parent/node['href']).resolve();links.append(f'<link id="{ident}" rel="stylesheet" href="{os.path.relpath(asset,HERE)}"'+(' disabled' if ident=='catalog-dark' else '')+'>')
html=(HERE/'catalog-template.html').read_text().replace('<!-- THEME -->','\n'.join(links)).replace('/* CATALOG_DATA */',json.dumps({'metadata':meta,'figures':rows},ensure_ascii=False).replace('</','<\\/'))
(HERE/'index.html').write_text(html)
print(json.dumps({'coverage':len(rows),'volume':meta['counts_by_volume'],'treatments':meta['treatments'],'first_wave':len(first_wave),'metadata_issues':sum(bool(r['metadata_issue']) for r in rows)},indent=2))

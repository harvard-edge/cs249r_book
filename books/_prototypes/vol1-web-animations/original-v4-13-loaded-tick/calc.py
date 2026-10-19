#!/usr/bin/env python3
"""Export canonical chapter LEGO arithmetic; no browser hardware registry."""
import re,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT/'mlsysim'))
source=ROOT/'books/vol4/13_placement/13_placement.qmd'
namespace={}
for cell in re.findall(r'```\{python\}[^\n]*\n(.*?)\n```',source.read_text(),re.S):
 if any('class '+name+':' in cell for name in ['PlacementMachine', 'PlacementDramTiming', 'PlacementLoadedRead']):exec(compile(cell,str(source),'exec'),namespace)
P=namespace['PlacementMachine'];R=namespace['PlacementLoadedRead'];us=lambda v:float(v.to('us').magnitude)
rows=[dict(label='unloaded',wait=0,read=us(P.sense),compute=us(R.compute_after_sense))]
for name,n,q,r in [('sense allowance',R.backlogs_sense,R.queue,R.own_read),('declared WCET',R.backlogs_wcet,R.queue,R.own_read),('deadline allocation',R.backlogs_deadline,R.queue,R.own_read),('one channel',R.backlogs_wcet_1ch,R.queue_1ch,R.own_read_1ch)]:rows.append(dict(label=name,wait=n*us(q),read=us(r),compute=us(R.compute_after_sense)))
data=dict(rows=rows,wcet_us=us(P.wcet),deadline_us=us(P.deadline),reserve_us=us(P.b_plus_i),period_us=us(P.tick))
if __name__=='__main__':
 out=Path(__file__).with_name('data.json');out.write_text(json.dumps(data,indent=2)+'\n');print(json.dumps(data,indent=2))

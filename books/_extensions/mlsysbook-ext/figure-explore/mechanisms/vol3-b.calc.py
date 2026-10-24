#!/usr/bin/env python3
"""Read-only verification of declared toy mechanisms; no hardware registry."""
import json,pathlib
p=pathlib.Path(__file__).with_name('vol3-b.json')
if not p.exists(): p=pathlib.Path('/tmp/vol3-b.json')
lessons=json.loads(p.read_text())
BY={x['key'].split(':',1)[1]:x for x in lessons}
def cases(l):
 yield 'original',l['steps']
 for v in l.get('variants',[]):yield v['id'],v['steps']
def metric(s,k):return next(m['value'] for m in s['snapshot']['metrics'] if m['id']==k)
def states(s):return s['snapshot']['objects']
def occupied(s,obj):return sum(x is not None for x in states(s)[obj]['slots'])
def nonnull(s,obj):return sum(x is not None for x in states(s)[obj]['values'])
def verify_metrics(key,mapping):
 l=BY[key]
 for case,steps in cases(l):
  for s in steps:
   for m,(o,kind) in mapping.items():
    expected=occupied(s,o) if kind=='pool' else nonnull(s,o)
    assert metric(s,m)==expected,(key,case,s['id'],m,expected)
verify_metrics('fig-vol3-index-invalidation',{'indexed':('index','pool'),'staged':('staging','pool')})
verify_metrics('fig-vol3-idempotency-flow',{'records':('record','pool')})
verify_metrics('fig-vol3-sandbox-pool-lifecycle',{'ready':('standby','pool'),'authority':('authority','pool'),'work':('work','pool')})
verify_metrics('fig-vol3-workspace-overlay-visibility',{'upper':('upperA','pool')})
verify_metrics('fig-vol3-checkpoint-consistent-cut',{'published':('published','pool'),'staging':('staging','pool')})
verify_metrics('fig-vol3-intent-before-dispatch',{'events':('log','sequence'),'reserved':('reservation','pool')})
# Primitive input: one transaction carries the only endpoint effect.
for case,steps in cases(BY['fig-vol3-idempotency-flow']):
 committed=[]
 for s in steps:
  if s['id']=='commit':committed.append('op-A')
  expected=len(set(committed))
  assert states(s)['effect']['value']==expected
  assert metric(s,'effects')==expected
# Primitive ASCII arrivals and fixed capacities; recompute all retained bytes.
stream=[('out','A'),('err','b'),('out','C'),('err','d'),('out','E'),('err','f')]
head_capacity=2;tail_capacity=3
for case,steps in cases(BY['fig-vol3-stream-ingestion']):
 for s in steps:
  q=states(s);n=nonnull(s,'arrival');arrivals=stream[:n]
  assert q['arrival']['values']==[c+':'+b for c,b in arrivals]+[None]*(len(stream)-n)
  raw=[b for c,b in arrivals];head=raw[:head_capacity];tail=raw[head_capacity:][-tail_capacity:]
  omitted=max(0,len(raw)-len(head)-len(tail));retained=len(head)+len(tail)
  assert q['head']['values']==head+[None]*(head_capacity-len(head))
  assert q['tail']['values']==tail+[None]*(tail_capacity-len(tail))
  assert q['omitted']['value']==omitted
  assert metric(s,'omitted')==omitted and metric(s,'drained')==n and metric(s,'retained')==retained
  assert n==retained+omitted
  if s['id']=='stage':assert q['staged']['values']==['Exited: code 1','Head: '+''.join(head),str(omitted)+' bytes omitted','Tail: '+''.join(tail)]
# Deterministic event fold: literal event names are inputs, not inferred outcomes.
log=['TaskStarted','Authorized test-r18','Reserved test-r18','Intent test-r18'];applied=[]
for case,steps in cases(BY['fig-vol3-event-sourcing']):
 for s in steps:
  if s['id'] in ['start','authorize','reserve','intent']:applied.append(log[len(applied)])
  q=states(s);assert q['log']['values']==log
  assert metric(s,'folded')==len(applied)
  tool_calls=[];assert q['calls']['value']==len(tool_calls) and metric(s,'calls')==len(tool_calls)
  if applied:
   expected=['Task active',None,None]
   if len(applied)>log.index('Authorized test-r18'):expected[0]='Task admitted'
   if len(applied)>log.index('Reserved test-r18'):expected[1]='Reservation held'
   if len(applied)>log.index('Intent test-r18'):expected[2]='test-r18 unresolved'
   assert q['recovery']['values']==expected
# File-layer model: derive merged view by private lookup then shared fallback.
for case,steps in cases(BY['fig-vol3-workspace-overlay-visibility']):
 baseline=['code:old','data:base']
 for s in steps:
  q=states(s);assert q['lower']['values']==baseline
  for suffix in ['A','B']:
   merged=[]
   for base,upper in zip(baseline,q['upper'+suffix]['slots']):
    merged.append(base if upper is None else None if upper['value'].startswith('whiteout:') else upper['value'])
   assert q['view'+suffix]['values']==merged
# Ownership/dependency guards apply to the full authored snapshots.
for case,steps in cases(BY['fig-vol3-sandbox-pool-lifecycle']):
 for s in steps:
  if occupied(s,'standby'):
   assert not occupied(s,'authority') and not occupied(s,'work') and not occupied(s,'workspace') and not occupied(s,'leased')
  if states(s)['exports']['values'][0] is not None:assert not occupied(s,'work')
for case,steps in cases(BY['fig-vol3-checkpoint-consistent-cut']):
 for s in steps:
  q=states(s)
  suffix=q['suffix']['values']
  assert suffix.index('Reserve test-r18') < suffix.index('Intent test-r18')
  if q['manifest']['value']=='c1':assert q['published']['slots'][1] is not None
  assert q['published']['slots'][0] is not None
for case,steps in cases(BY['fig-vol3-intent-before-dispatch']):
 for s in steps:
  q=states(s)
  if q['request']['values'][0] is not None:
   assert q['worker']['value']=='Intent acknowledged' and 'ToolIntent test-r18' in q['log']['values']
  if s['id']=='pending':assert q['evidence']['values'][1]=='Outcome unresolved'
# No physical latency, speed, ratios, or uncomputed arithmetic are displayed.
for l in lessons:
 for case,steps in cases(l):
  last=steps[-1]
  print(l['key'],case,last['id'],json.dumps({m['id']:m['value'] for m in last['snapshot']['metrics']},ensure_ascii=False))
print('PASS: byte retention, event folds, effect counts, occupancy, overlay lookup, and causal guards.')

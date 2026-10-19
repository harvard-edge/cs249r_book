#!/usr/bin/env python3
"""Recompute declared lesson examples; check mode never writes files."""
import copy,json,math,pathlib,sys

def obj(i,k,label,role,config,n):
 return dict(id=i,kind=k,label=label,role=role,config=config,layout={'wide':{'row':n//2,'column':n%2},'narrow':{'row':n,'column':0}})
def seq(i,label,n,slots,role='reference',labels=None):
 c={'slots':slots}
 if labels:c['slotLabels']=labels
 return obj(i,'sequence',label,role,c,n)
def reg(i,label,n,typ='State',unit=None):
 c={'type':typ}
 if unit:c['unit']=unit
 return obj(i,'register',label,'control',c,n)
def pool(i,label,n,slots):return obj(i,'pool',label,'memory',{'slots':slots,'columns':min(slots,4)},n)
def mat(i,label,n,r,c,rl,cl):return obj(i,'matrix',label,'reference',{'rows':r,'columns':c,'rowLabels':rl,'columnLabels':cl},n)
def ss(v,status=''):return {'values':v,'status':status}
def rs(v,status=''):return {'value':v,'status':status}
def ps(v,status=''):return {'slots':v,'status':status}
def slot(v,owner,status='retained'):return {'value':v,'owner':owner,'status':status}
def metric(i,label,value,unit):return {'id':i,'label':label,'value':value,'unit':unit}
def lesson(key,title,q,scope,family,objects,invariants):return dict(schema=1,key='vol3:fig-vol3-'+key,title=title,question=q,scope=scope,family=family,objects=objects,steps=[],invariants=invariants)
def step(l,i,title,ex,states,metrics,result,kind=None,subjects=None,from_=None,to=None,tokens=None):
 tr=None if not l['steps'] else {'kind':kind,'subjects':subjects or [],'meaning':ex}
 if tr is not None:
  if from_:tr['from']=from_
  if to:tr['to']=to
 l['steps'].append(dict(id=i,title=title,explanation=ex,snapshot=dict(objects=copy.deepcopy(states),tokens=tokens or [],metrics=metrics,activeConnections=[],result=result),transition=tr))
def build():
 out=[]
 # A durable obligation exists before either transport result is known.
 l=lesson('saga-flow','An owned duty survives a lost reply','When can cleanup ownership safely end?','The figure’s environment operation, with test termination and durable artifact preservation already verified. Financial settlement is outside this physical cleanup example.','ownership',[pool('duties','Owned cleanup duties',0,1),mat('forward','Independent operation records',1,2,2,['provision','cleanup'],['logical operation','outcome']),seq('proof','Retained outcome evidence',2,2,labels=['provision evidence','release evidence']),seq('preconditions','Verified cleanup predecessors',3,2)],['The cleanup duty is registered before provisioning dispatch.','Unknown outcomes preserve the duty and its owner.','Only correlated release evidence closes the physical cleanup duty; audit history remains.'])
 s={'duties':ps([None]),'forward':ss(['provision_test_env_r18','not dispatched','release_test_env_r18','not dispatched']),'proof':ss([None,None]),'preconditions':ss(['test ended','artifacts durable'])}
 events=[('ready','Before registration','No request has left the runtime.',None,None),('register','Register the cleanup duty','The coordinator records an unbound duty before dispatch.','write',lambda:s['duties']['slots'].__setitem__(0,slot('release: target unbound','coordinator'))),('dispatch','Dispatch provisioning','The recorded duty backs the provisioning request.','write',lambda:s['forward']['values'].__setitem__(1,'in flight')),('lost','Record the lost provisioning reply','Transport termination leaves creation unresolved.','write',lambda:s['forward']['values'].__setitem__(1,'unknown: reply lost')),('bind','Bind the duty from correlated evidence','An authenticated result identifies the environment; a lookup guess is insufficient.','write',lambda:(s['proof']['values'].__setitem__(0,'provision → env_r18_01'),s['duties']['slots'].__setitem__(0,slot('release env_r18_01','coordinator')),s['forward']['values'].__setitem__(1,'created: evidence confirmed'))),('release','Dispatch eligible cleanup','Both predecessors hold and the duty has a bound target.','write',lambda:s['forward']['values'].__setitem__(3,'in flight')),('release-timeout','Record an unconfirmed release','The cleanup reply is lost; the environment may be gone or still allocated.','write',lambda:s['forward']['values'].__setitem__(3,'unknown: reply lost')),('close','Close on authenticated release evidence','Correlated evidence establishes the declared deallocation postcondition.','release',lambda:(s['proof']['values'].__setitem__(1,'env_r18_01: not allocated'),s['duties']['slots'].__setitem__(0,None),s['forward']['values'].__setitem__(3,'deallocated: evidence confirmed')))]
 for i,t,e,k,fn in events:
  if fn:fn()
  count=sum(x is not None for x in s['duties']['slots'])
  step(l,i,t,e,s,[metric('open','Owned open duties',count,'duties')],f'{count} open cleanup duties; evidence is retained independently.',k,['duties' if i in ('register','bind','close') else 'forward'])
 l['checks']=[{'kind':'metricEqualsOccupied','metric':'open','object':'duties'}];out.append(l)
 # Exact intervals from the chapter, never add enclosing and child durations.
 intervals=[('retrieve',0,0,.2,'Retrieve'),('plan',0,.2,3.2,'Plan'),('tests',1,3.2,9.2,'Tests'),('lint',2,3.2,5.2,'Lint'),('synthesis',0,9.7,11.7,'Synthesis')]
 l=lesson('overlapping-trace-intervals','A join waits for its longest branch','Why does early lint completion leave synthesis blocked?','Exact hypothetical host-clock intervals from the figure, in seconds. The client timeout and callback are a separate operation, with no clock mapping into these intervals. No task acceptance claim follows.','schedule',[obj('trace','lanes','Observed host-clock spans','compute',{'lanes':['Context/model','Tests','Lint'],'start':0,'end':12,'axisLabel':'Host elapsed time (s)','ticks':[0,3,6,9,12]},0),reg('wall','Enclosing trajectory duration',1,'Duration','s'),seq('operation','Separate linked operation',2,2,labels=['client span','operation outcome'])],['Recorded span geometry stays fixed.','Synthesis cannot start until both tests and lint finish; the gap is unattributed.','Client timeout ends a span without establishing the external outcome.'])
 states=['done','done','running','running','blocked'];s={'trace':{'segments':[],'cursor':3.2,'status':''},'wall':rs(intervals[-1][3]-intervals[0][2]),'operation':ss(['open','pending'])}
 for i,t,e,c,updates,op in [('parallel','Checks run concurrently','Retrieval and planning are complete; both independent checks are running.',3.2,{},None),('lint-done','Complete lint','Lint finishes while tests continue to hold the join.',5.2,{3:'done'},None),('tests-done','Complete tests','The longer branch finishes; synthesis has not started.',9.2,{2:'done'},None),('synthesis-start','Start synthesis after the observed gap','The host records a later start; the intervening interval has no attribution.',9.7,{4:'running'},None),('response','Complete synthesis','The response closes the enclosing observation window.',11.7,{4:'done'},None),('timeout','End the separate client span on timeout','A timeout settles the span boundary, leaving the operation unresolved.',11.7,{},['timeout','unresolved']),('callback','Correlate the late operation result','The linked callback settles the operation outcome; it does not prove task acceptance.',11.7,{},['timeout','effect confirmed'])]:
  for j,v in updates.items():states[j]=v
  s['trace']={'segments':[dict(id=id,lane=lane,start=a,end=b,label=label,state=states[j],role='compute') for j,(id,lane,a,b,label) in enumerate(intervals)],'cursor':c,'status':'Observed intervals; no benchmark claim'}
  if op:s['operation']=ss(op)
  leaf=sum(b-a for _,_,a,b,_ in intervals);gap=intervals[-1][2]-max(intervals[2][3],intervals[3][3])
  step(l,i,t,e,s,[metric('leafsum','Leaf-duration sum',round(leaf,1),'s'),metric('wall','Enclosing elapsed',s['wall']['value'],'s'),metric('gap','Unattributed interval',round(gap,1),'s')],f'Elapsed {s["wall"]["value"]} s; overlapping leaf durations total {round(leaf,1)} s.', 'write' if op else 'execute',['operation' if op else 'trace'])
 out.append(l)
 # Independent population accounting survives diagnostic body loss.
 l=lesson('tail-sampling-pipeline','Body loss does not erase a trial','What survives when a verdict arrives after body eviction?','Two admitted toy trials A and B; two bounded body slots. Permitted minimized bodies only. Selection is intentionally diagnostic, with no claim that the retained set estimates population rates.','ownership',[mat('account','Complete admitted-trial accounting',0,2,3,['A','B'],['verdict','body record','selection']),pool('buffer','Bounded provisional trace bodies',1,2),pool('store','Retained diagnostic traces',2,2)],['Every admitted trial stays in the accounting table after its body disappears.','A late verdict cannot reconstruct an evicted body.','Retention policy and inclusion record accompany retained diagnostics.'])
 s={'account':ss(['pending','not captured','undecided','pending','not captured','undecided']),'buffer':ps([None,None]),'store':ps([None,None])}
 ev=[('admitted','Record every admitted trial','The denominator is recorded before diagnostic retention.',None,None),('capture','Capture a permitted minimized batch','Both bodies enter bounded staging; accounting is independent.','write',lambda:(s['buffer']['slots'].__setitem__(slice(None),[slot('A body','trial A'),slot('B body','trial B')]),s['account']['values'].__setitem__(1,'buffered'),s['account']['values'].__setitem__(4,'buffered'))),('evict','Evict A before its verdict','Pressure removes A’s body and records the diagnostic loss.','release',lambda:(s['buffer']['slots'].__setitem__(0,None),s['account']['values'].__setitem__(1,'evicted'),s['account']['values'].__setitem__(2,'body unavailable'))),('late-A','Record A’s late failed verdict','A remains a failed admitted trial even though its detailed body is gone.','write',lambda:s['account']['values'].__setitem__(0,'failed')),('verdict-B','Record B’s failed verdict','B still has its permitted body and receives high diagnostic priority.','write',lambda:s['account']['values'].__setitem__(3,'failed')),('retain-B','Copy B into retained storage','The eligible body is copied with its inclusion record; staging survives the store write.','copy',lambda:(s['store']['slots'].__setitem__(0,slot('B body; policy P1; inclusion 1','trial B')),s['account']['values'].__setitem__(4,'retained; staging held'),s['account']['values'].__setitem__(5,'P1; inclusion 1'))),('retire-staging','Release B’s consumed staging slot','The retained store write is complete, so its staging source can now be released.','release',lambda:(s['buffer']['slots'].__setitem__(1,None),s['account']['values'].__setitem__(4,'retained')))]
 for i,t,e,k,fn in ev:
  if fn:fn()
  a=s['account']['values']; n=len(a)//3;ret=sum(x is not None for x in s['store']['slots']);failed=sum(a[j]=='failed' for j in range(0,len(a),3));lost=sum(a[j]=='evicted' for j in range(1,len(a),3))
  step(l,i,t,e,s,[metric('cohort','Admitted denominator',n,'trials'),metric('failed','Known failed trials',failed,'trials'),metric('retained','Retained bodies',ret,'bodies'),metric('lost','Recorded evictions',lost,'bodies')],f'Accounting retains {n} trials; diagnostic storage retains {ret} bodies.',k,['store','buffer'] if i=='retain-B' else ['account','buffer'],from_={'object':'buffer','index':1} if i=='retain-B' else None,to={'object':'store','index':0} if i=='retain-B' else None,tokens=[dict(id='B-retained-copy',label='B minimized body',value='B body',role='memory',owner='trial B',at={'object':'store','index':0},state='copy',sourceAt={'object':'buffer','index':1})] if i=='retain-B' else [])
 out.append(l)
 # Credits reserve the entire run lifetime, including intermediate ownership.
 l=lesson('collection-pipeline','One credit spans all four responsibilities','Why does a finished rollout still hold collection capacity?','Two toy credits and two admitted runs. Each credit covers one fixed payload ceiling; execution resources have separate bounds. The hard limit is demonstrated, not a quantitative watermark controller.','ownership',[pool('credits','Collection reservations',0,2),mat('pipeline','Records across responsibilities',1,2,4,['A','B'],['rollout','capture','staging','assessment']),seq('terminal','Durable dispositions',2,2,labels=['A','B']),reg('admission','Admission gate',3)],['Every nonterminal admitted run keeps a collection reservation across all stages.','Rollout completion alone returns no credit.','A credit returns only after a durable disposition is confirmed.'])
 s={'credits':ps([None,None]),'pipeline':ss([None]*8),'terminal':ss([None,None]),'admission':rs('open')}
 def loc(col):
  s['pipeline']['values']=[('A' if r==0 else 'B') if c==col else None for r in range(2) for c in range(4)]
 events=[('empty','Before admission','The full collection capacity is available.',None,None),('admit','Admit a bounded batch','Atomic admission reserves one collection credit per run.','write',lambda:(s['credits']['slots'].__setitem__(slice(None),[slot('credit A','run A'),slot('credit B','run B')]),loc(0))),('capture','Capture both completed rollouts','This parallel capture retains each run’s existing reservation.','move',lambda:loc(1)),('stage','Stage both captured records','Moving responsibility changes the holding location without freeing capacity.','move',lambda:loc(2)),('assess','Begin downstream assessment','Both runs occupy assessment while their reservations remain held.','move',lambda:loc(3)),('blocked','Reject an additional admission','No collection reservation remains; the hard cap stops a new run.','write',lambda:s['admission'].update(value='C denied: no credit')),('durable','Confirm A’s durable disposition','A leaves pending assessment; its credit remains held until the return is recorded.','write',lambda:(s['pipeline']['values'].__setitem__(3,None),s['terminal']['values'].__setitem__(0,'A: durable diagnostic disposition'))),('return','Return A’s collection credit','Confirmed durable disposition authorizes this capacity release.','release',lambda:(s['credits']['slots'].__setitem__(0,None),s['admission'].update(value='one admission available')))]
 for i,t,e,k,fn in events:
  if fn:fn()
  held=sum(x is not None for x in s['credits']['slots']);free=len(s['credits']['slots'])-held
  step(l,i,t,e,s,[metric('held','Held credits',held,'credits'),metric('free','Available credits',free,'credits')],f'{held} held + {free} available = {len(s["credits"]["slots"])} total credits.',k,['credits','pipeline','terminal','admission'])
 l['checks']=[{'kind':'metricEqualsOccupied','metric':'held','object':'credits'}];out.append(l)
 # Packed buffer isolation is a score operation, not a speedup assertion.
 packed=['A0','A1','B0','B1']; q=[1,0]; keys=[[1,0],[2,0],[3,0],[4,0]];values=[10,20,30,40];query=2;boundary=2
 raw=[sum(a*b for a,b in zip(q,k)) for k in keys]
 l=lesson('block-diagonal-mask','Leaked context changes the attention output','What contribution disappears when example B cannot read example A?','Toy packed buffer with two tokens per example, one two-component B0 query, and scalar values. Dot products are unscaled toy logits. This score-row example makes no claim about skipped kernel work.','spatial',[seq('packed','Shared buffer; separate examples',0,len(packed),'input'),mat('vectors','Query and key vectors',1,len(keys)+1,len(q),['q B0']+packed,['component 0','component 1']),seq('scores','Pre-softmax score row for B0',2,len(keys),'compute',packed),seq('weights','Attention weights for B0',3,len(keys),'output',packed),seq('values','Scalar values by key position',4,len(values),'input',packed),seq('output','Weighted attention result',5,2,'output',['ordinary causal','example isolated'])],['Packed inputs, query, keys, and values remain unchanged.','Causality excludes B1; membership separately excludes A0 and A1 before softmax.','The ordinary output includes value contributions from example A; the isolated output cannot.','Displayed weights are rounded for legibility; outputs use unrounded probabilities.'])
 s={'packed':ss(packed),'vectors':ss(q+[x for row in keys for x in row]),'scores':ss([None]*len(keys)),'weights':ss([None]*len(keys)),'values':ss(values),'output':ss([None,None])}
 exact_weights=None;leak='not produced'
 cases=[('inputs','Inspect independent packed examples','B0 is the first query of the second example.',None),('dot','Compute the query–key score row','Dot products produce candidate scores for the selected query.','raw'),('causal','Apply ordinary causality','Future B1 is excluded, but A0 and A1 retain finite scores.','causal'),('ordinary-softmax','Normalize ordinary causal scores','The first example receives nonzero probability mass.','ordinary-softmax'),('ordinary-output','Reduce values under ordinary causality','Cross-example values now contribute to B0’s attention result.','ordinary-output'),('boundary','Mask cross-example scores before softmax','A0 and A1 become −∞; old weights are invalidated before recomputation.','block'),('isolated-softmax','Normalize the example-isolated scores','Only B0 remains finite, giving its value all probability mass.','isolated-softmax'),('isolated-output','Reduce values within the example','The isolated output keeps B0’s contribution and removes the first example’s contribution.','isolated-output')]
 for i,t,e,action in cases:
  if action=='raw':s['scores']=ss(raw)
  if action=='causal':s['scores']=ss([v if j<=query else '−∞' for j,v in enumerate(raw)])
  if action=='block':s['scores']=ss([v if boundary<=j<=query else '−∞' for j,v in enumerate(raw)]);s['weights']=ss([None]*len(keys));exact_weights=None
  if action in ('ordinary-softmax','isolated-softmax'):
   finite=[x for x in s['scores']['values'] if isinstance(x,(int,float))];m=max(finite);den=sum(math.exp(x-m) for x in finite)
   exact_weights=[math.exp(v-m)/den if isinstance(v,(int,float)) else 0 for v in s['scores']['values']]
   s['weights']=ss([round(w,4) for w in exact_weights])
  if action in ('ordinary-output','isolated-output'):
   result=sum(w*v for w,v in zip(exact_weights,values));index=0 if action=='ordinary-output' else 1;s['output']['values'][index]=round(result,4)
   if index==0:leak=round(sum(exact_weights[j]*values[j] for j in range(boundary)),4)
  visible=sum(isinstance(x,(int,float)) for x in s['scores']['values']);cross=sum(isinstance(s['scores']['values'][j],(int,float)) for j in range(boundary))
  step(l,i,t,e,s,[metric('finite','Finite scores',visible,'keys'),metric('cross','Finite cross-example scores',cross,'keys'),metric('leak','Ordinary output contribution from A',leak,'value units')],f'{visible} finite scores; {cross} cross-example scores. Both completed outputs remain visible.', 'execute' if action in ('raw','ordinary-softmax','isolated-softmax','ordinary-output','isolated-output') else 'transform',['output' if action in ('ordinary-output','isolated-output') else 'weights' if action in ('ordinary-softmax','isolated-softmax') else 'scores'])
 l['checks']=[{'kind':'constantValues','object':id} for id in ['packed','vectors','values']];out.append(l)
 # Behavior versions persist independently of execution slots and verdicts.
 l=lesson('asynchronous-freshness-pipeline','A passed record can still be stale','How does trainer progress change eligibility without changing a verdict?','Focused toy admission case for the figure’s asynchronous path: three records sampled at versions 2, 1, and 0, trainer initially 3, maximum lag 2. It does not reproduce the four-member synchronous timing example or establish complete group eligibility.','schedule',[pool('workers','Live rollout environment slots',0,3),mat('records','Bounded record buffer',1,3,3,['A','B','C'],['behavior version','verdict','admission']),reg('trainer','Current trainer version',2,'Version'),seq('ledger','Recorded exclusion reasons',3,3,labels=['A','B','C'])],['Each record keeps the version that generated it.','A rollout completion frees an environment slot while its record remains buffered.','Verified rewards, lag eligibility, and complete cohort eligibility are separate checks.'])
 behavior=[2,1,0];bound=2;s={'workers':ps([slot(x,'rollout '+x,'allocated') for x in ['A','B','C']]),'records':ss([v for b in behavior for v in [b,'pending','not checked']]),'trainer':rs(3),'ledger':ss([None]*3)}
 def complete(j):s['workers']['slots'][j]=None;s['records']['values'][j*3+1]='pass'
 def decide(j):
  lag=s['trainer']['value']-behavior[j];s['records']['values'][j*3+2]='record eligible' if lag<=bound else 'excluded';s['ledger']['values'][j]=f'lag {lag} ≤ {bound}' if lag<=bound else f'lag {lag} > {bound}';return lag
 events=[('running','Rollouts run under fixed behavior versions','The trainer is already ahead of these in-flight records.',None,None),('A-complete','Complete and verify A','A’s environment slot frees; its compact verified record stays in the bounded buffer.','execute',lambda:complete(0)),('A-admit','Check A at the current trainer version','A passes the record-level lag gate; group closure remains a separate obligation.','write',lambda:decide(0)),('trainer-update','Commit one trainer update','The trainer advances while B and C retain their original behavior versions.','execute',lambda:s['trainer'].update(value=s['trainer']['value']+1)),('B-complete','Complete and verify B','B has a pass verdict, independently of its current lag.','execute',lambda:complete(1)),('B-exclude','Exclude B by the lag bound','The verified record exceeds the version bound; its reason remains recorded.','write',lambda:decide(1)),('C-complete','Complete and verify C','The final environment slot frees; the older record still has its generating version.','execute',lambda:complete(2)),('C-exclude','Exclude C by the lag bound','A pass verdict does not override the recorded freshness limit.','write',lambda:decide(2))]
 for i,t,e,k,fn in events:
  if fn:fn()
  v=s['trainer']['value'];live=sum(x is not None for x in s['workers']['slots']);excluded=sum(x=='excluded' for x in s['records']['values'])
  step(l,i,t,e,s,[metric('live','Live environment slots',live,'slots'),metric('lagA','A lag at current trainer',v-behavior[0],'updates'),metric('lagB','B lag at current trainer',v-behavior[1],'updates'),metric('lagC','C lag at current trainer',v-behavior[2],'updates'),metric('excluded','Recorded exclusions',excluded,'records')],f'{live} environment slots remain live; {excluded} records have recorded freshness exclusions.',k,['workers','records','trainer','ledger'])
 l['checks']=[{'kind':'metricEqualsOccupied','metric':'live','object':'workers'}];out.append(l)
 # Publication checks are bound to one exact candidate and captured head.
 l=lesson('occ-worktrees','A checked candidate still needs a head guard','Why can a clean and tested candidate fail publication?','Symbolic repository example using disjoint interface and caller files. Hash labels identify exact states rather than real commits. A target-head change invalidates the publication attempt; no external effects are rolled back.','control',[seq('worker','Private caller workspace',0,2,labels=['base','required interface']),seq('target','Shared target ref',1,2,labels=['head','interface']),seq('candidate','Private integration candidate',2,3,labels=['identity','captured head','required interface']),seq('evidence','Evidence bound to the candidate',3,2,labels=['dependency check','exact candidate test']),reg('publication','Publication result',4)],['Textual merge success does not establish semantic dependency compatibility.','Tests name the exact integration candidate they evaluated.','Atomic publication requires the captured head and all publication guards to hold.'])
 s={'worker':ss(['h0','v1']),'target':ss(['h0','v1']),'candidate':ss([None]*3),'evidence':ss([None]*2),'publication':rs('private')}
 events=[('fork','Begin with private caller work','The caller depends on interface v1 at the common base.',None,None),('interface','Publish an independent interface change','The target interface becomes v2 while the private caller still requires v1.','write',lambda:s['target'].update(values=['h1','v2'])),('merge','Build a textually clean candidate','Disjoint file edits merge, but the combined candidate retains a stale dependency.','write',lambda:s['candidate'].update(values=['c1','h1','v1'])),('revise','Revise the private caller dependency','The worker adapts its caller to the current interface; old candidate c1 is unchanged.','write',lambda:s['worker'].update(values=['h0','v2'])),('rebuild','Build a revised integration candidate','Candidate c2 captures h1 and carries the revised caller.','write',lambda:s['candidate'].update(values=['c2','h1','v2'])),('check','Validate and test exact candidate c2','One bounded validation job checks declared dependencies and executes the required candidate tests.','execute',lambda:s['evidence'].update(values=['c2 vs h1: supported','c2: required tests pass'])),('head-move','Advance the target before publication','An intervening accepted update changes the target from captured head h1.','write',lambda:s['target'].update(values=['h2','v3'])),('guard','Attempt the expected-head update','The atomic guard observes h2 rather than h1 and refuses publication.','execute',lambda:s['publication'].update(value='head moved: revalidate'))]
 for i,t,e,k,fn in events:
  if fn:fn()
  compatible=s['candidate']['values'][2] is not None and s['candidate']['values'][2]==s['target']['values'][1];matches=s['candidate']['values'][1] is not None and s['candidate']['values'][1]==s['target']['values'][0]
  step(l,i,t,e,s,[metric('compatible','Current dependency match',int(compatible),'boolean'),metric('head','Captured head still current',int(matches),'boolean')],'Target remains '+s['target']['values'][0]+'; publication state: '+s['publication']['value']+'.',k,['publication' if i=='guard' else 'evidence' if i=='check' else 'worker' if i=='revise' else 'target' if i in ('interface','head-move') else 'candidate'])
 out.append(l)
 # Exact integer milli-dollar accounting from the chapter’s declared scenario.
 total=5000;planning=42;trace_alloc=1200;fuzz_alloc=2500;trace_charge=380;fuzz_charge=850;pending=1000;late=700
 l=lesson('hierarchical-budget','Stopping preserves the child’s backing','Which event allows the root to reuse a canceled child’s funds?','The chapter’s hypothetical budget records in integer milli-dollars, not provider prices. Starts after both child allocations and local charges are recorded. The child view and root reservation are the same backing seen at two levels.','ownership',[reg('settled','Root settled charges',0,'Budget','milli-dollars'),reg('reserved','Root reserved backing',1,'Budget','milli-dollars'),reg('free','Root free balance',2,'Budget','milli-dollars'),obj('partition','lanes','Conserved account partitions','reference',{'lanes':['Root','Fuzz child'],'start':0,'end':total,'axisLabel':'Milli-dollars within each account','ticks':[0,1000,2500,5000]},3),mat('child','Fuzz child account',4,1,3,['Fuzz'],['settled','remote reserved','free']),reg('admission','Fuzz child admission',5)],['At every node, settled plus reserved plus free equals its own allocation.','A parent reservation and the backed child account must not be added together as spending.','Stopping closes admission but releases no remote liability.','Only authenticated final settlement, followed by child closure, returns uncommitted funds to the root.'])
 E=planning;R=trace_alloc+fuzz_alloc;F=total-E-R;ce=fuzz_charge;cr=pending;cf=fuzz_alloc-ce-cr;gate='open'
 states=[]
 states.append(('allocated','Both allocations remain backed','The trace and fuzz children have local charge records; root reserves their complete allocations.',None,E,R,F,ce,cr,cf,gate))
 E+=trace_charge;R-=trace_alloc;F+=trace_alloc-trace_charge
 states.append(('trace-close','Close the fully settled trace child','The trace child returns only its unused allocation and posts its settled charge to the root.','write',E,R,F,ce,cr,cf,gate))
 gate='stopped: no new admission';states.append(('stop','Stop the fuzz child locally','New spending admission closes; the remote reservation and whole root backing remain.','write',E,R,F,ce,cr,cf,gate))
 ce+=late;cf+=cr-late;cr-=pending
 states.append(('settle','Settle the authenticated late bill','The child consumes its bounded remote reservation; the root still backs the open child.','write',E,R,F,ce,cr,cf,gate))
 E+=ce;R-=fuzz_alloc;F+=cf;gate='closed'
 states.append(('close','Close the fully settled fuzz child','No remote reservation remains; root imports the final charge and returns only unused funds.','release',E,R,F,ce,cr,cf,gate))
 for i,t,e,k,E,R,F,ce,cr,cf,gate in states:
  assert E+R+F==total and ce+cr+cf==fuzz_alloc and min(E,R,F,ce,cr,cf)>=0
  segments=[]
  for lane,amounts in enumerate(((E,R,F),(ce,cr,cf))):
   start=0
   for ident,label,amount,role in zip(['settled','reserved','free'],['Settled','Reserved','Free'],amounts,['reference','memory','output']):
    end=start+amount
    if amount:segments.append(dict(id=f'account-{lane}-{ident}',lane=lane,start=start,end=end,label=f'{label} {amount}',state='done',role=role))
    start=end
  s={'settled':rs(E),'reserved':rs(R),'free':rs(F),'partition':{'segments':segments,'status':'Balance partition geometry changes with account transfers'},'child':ss([ce,cr,cf]),'admission':rs(gate)}
  step(l,i,t,e,s,[metric('rootTotal','Root conserved total',E+R+F,'milli-dollars'),metric('childTotal','Child allocation',ce+cr+cf,'milli-dollars'),metric('pending','Child remote backing',cr,'milli-dollars')],f'Root: {E} settled + {R} reserved + {F} free = {total} milli-dollars.',k,['admission' if i=='stop' else 'child' if i=='settle' else 'settled','reserved','free','partition'])
 out.append(l)
 results={
 'saga-flow':[
 'Provisioning has not started, and no cleanup duty exists yet.',
 'The coordinator now owns an unbound cleanup duty before any external effect.',
 'Provisioning is in flight; the already-recorded cleanup duty survives a lost response.',
 'Creation is unresolved. The coordinator keeps the duty without guessing its target.',
 'Correlated evidence binds env_r18_01 to the existing duty; ownership remains with the coordinator.',
 'Cleanup is in flight against the verified target after both prerequisites are satisfied.',
 'Release remains unconfirmed. The duty stays owned even though its request timed out.',
 'Verified deallocation closes the physical duty; provisioning and release evidence remain visible.'
 ],
 'overlapping-trace-intervals':[
 'Both independent checks run after planning; synthesis waits at their join.',
 'Lint is done, but unfinished tests still block synthesis.',
 'Both checks are complete. The observed synthesis start still lies beyond the unattributed gap.',
 'Synthesis begins at its recorded start; these spans do not explain the intervening gap.',
 'The response completes the enclosing interval. Overlapping child durations cannot be added as elapsed time.',
 'The separate client span has ended, but its external operation remains unresolved.',
 'The correlated callback confirms the external effect; whole-task acceptance still needs its own evidence.'
 ],
 'tail-sampling-pipeline':[
 'Both admitted trials exist in accounting before either diagnostic body is captured.',
 'The provisional bodies occupy bounded staging while their verdicts remain pending.',
 'A loses its diagnostic body, but its admitted-trial record and eviction flag persist.',
 'A is now a known failed trial with missing detail; the verdict cannot reconstruct its evicted body.',
 'B is a known failed trial whose body remains available for high-priority diagnostic retention.',
 'B now has a retained copy and an inclusion record; its staging source remains until the store write is complete.',
 'B’s staging slot is free after its final consumer. A’s missing body and both admitted-trial records remain visible.'
 ],
 'collection-pipeline':[
 'The reservation pool is free; no record is admitted.',
 'Both runs now own a collection reservation before rollout begins.',
 'Completed rollouts move into capture while keeping their existing reservations.',
 'The captured records move into staging; reservation ownership does not end at the stage boundary.',
 'Assessment holds both records and both reservations, even though rollout has finished.',
 'C cannot enter the pipeline because all collection capacity is still owned downstream.',
 'A has a confirmed durable disposition. Its reservation stays held until the explicit return is recorded.',
 'A’s returned credit can admit another run while B continues to hold its own reservation.'
 ],
 'block-diagonal-mask':[
 'B0 shares a physical buffer with A but belongs to a separate example.',
 'The unmasked score row contains both earlier-example keys and the future B1 key.',
 'Ordinary causality removes future B1 but leaves finite scores for A’s unrelated context.',
 'The ordinary softmax assigns nonzero weight to A’s values.',
 'The ordinary attention output now contains a computed contribution from the unrelated example A.',
 'Cross-example scores become −∞ before normalization. The old weights are invalid and disappear.',
 'Only B0 retains probability mass; A can no longer influence this query’s value reduction.',
 'The corrected result uses only B0’s value; the contaminated ordinary result remains alongside it for comparison.'
 ],
 'asynchronous-freshness-pipeline':[
 'The rollout records retain their generating versions while the trainer is already ahead.',
 'A’s environment slot is free, but its verified compact record stays buffered for admission.',
 'A satisfies the record-level lag gate. This does not close a planned training cohort.',
 'One trainer update increases the lag of every fixed behavior version without changing any verdict.',
 'B has a pass verdict and a free environment slot; freshness eligibility is still unchecked.',
 'B’s verified record is excluded as stale, and the reason remains in the ledger.',
 'C’s completed environment slot is free; its older generating version still belongs to the buffered record.',
 'C is also excluded by lag. A pass verdict does not override the freshness bound.'
 ],
 'occ-worktrees':[
 'The private caller depends on the original interface at the common base.',
 'The target interface changes while the private caller still expects the older contract.',
 'Candidate c1 merges without a text conflict but carries an unsupported caller dependency.',
 'The private caller now targets v2; candidate c1 remains unchanged and still needs rebuilding.',
 'Candidate c2 captures h1 with the revised caller; it needs evidence of its own.',
 'The required checks support exact candidate c2 against captured head h1.',
 'The target advances to h2 after c2 was checked; the old candidate evidence does not cover the new head.',
 'The expected-head update refuses publication and returns to revalidation; h2 remains the target.'
 ],
 'hierarchical-budget':[
 'The root reserves both complete child allocations while the fuzz child separately tracks its own pending remote liability.',
 'The settled trace child closes; only its unused funds return to the root’s free balance.',
 'The fuzz child stops admitting work. Its remote liability and complete root reservation remain unchanged.',
 'Authenticated final billing settles the child’s remote liability; the root still backs the open child account.',
 'The fully settled fuzz child closes. Its final charge reaches the root and only its unused funds become free.'
 ]}
 for l in out:
  suffix=l['key'].removeprefix('vol3:fig-vol3-')
  assert len(results[suffix])==len(l['steps'])
  for st,result in zip(l['steps'],results[suffix]):st['snapshot']['result']=result

 return out

def main():
 expected=build();path=pathlib.Path(__file__).with_suffix('').with_suffix('.json')
 if len(sys.argv)>1 and sys.argv[1]=='--emit':
  print(json.dumps(expected,ensure_ascii=False,indent=2));return
 actual=json.loads(path.read_text());assert actual==expected,'Authored JSON differs from recomputed lesson states'
 for l in actual:
  for st in l['steps']:
   print(l['key'],st['id'],', '.join(f'{m["id"]}={m["value"]} {m["unit"]}' for m in st['snapshot']['metrics']))
 print('PASS: all snapshots and metrics match explicit inputs and computed causal states; no files rewritten.')
if __name__=='__main__':main()

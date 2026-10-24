#!/usr/bin/env python3
"""Reproduce declared schematic lessons. Default mode verifies without writing."""
import copy,json,pathlib,sys

def obj(i,kind,label,role,config,row,col=0,span=1,narrow=None):
 return dict(id=i,kind=kind,label=label,role=role,config=config,layout={'wide':{'row':row,'column':col,'columnSpan':span},'narrow':{'row':narrow if narrow is not None else row*2+col,'column':0,'columnSpan':1}})
def seq(i,label,n,row,col=0,role='input',labels=None,narrow=None):
 c={'slots':n}
 if labels:c['slotLabels']=labels
 return obj(i,'sequence',label,role,c,row,col,narrow=narrow)
def pool(i,label,n,row,col=0,labels=None):
 c={'slots':n,'columns':min(n,4)}
 if labels:c['slotLabels']=labels
 return obj(i,'pool',label,'memory',c,row,col)
def reg(i,label,row,col=0,typ='State',unit=None):
 c={'type':typ}
 if unit:c['unit']=unit
 return obj(i,'register',label,'control',c,row,col)
def S(v,status=''):return {'values':v,'status':status}
def R(v,status='',previous=None):
 a={'value':v,'status':status}
 if previous is not None:a['previous']=previous
 return a
def P(v,owner='Cache',status=''):
 return {'slots':[None if x is None else {'value':x,'owner':owner,'status':'retained'} for x in v],'status':status}
def metric(i,label,value,unit='items'):return dict(id=i,label=label,value=value,unit=unit)
def step(i,title,exp,states,metrics,result,kind=None,subjects=None,fr=None,to=None,tokens=None):
 tr=None if kind is None else dict(kind=kind,subjects=subjects or [],meaning=exp)
 if tr is not None:
  if fr:tr['from']=fr
  if to:tr['to']=to
 return dict(id=i,title=title,explanation=exp,snapshot=dict(objects=copy.deepcopy(states),tokens=tokens or [],metrics=metrics,activeConnections=[],result=result),transition=tr)
def lesson(key,title,question,scope,family,objects,steps,invariants,**kw):
 return dict(schema=1,key='vol3:'+key,title=title,question=question,scope=scope,family=family,objects=objects,steps=steps,invariants=invariants,**kw)
def nonnull(x):return sum(v is not None for v in x)

def first_token():
 # Event coordinates are declared schematic inputs, not measured milliseconds.
 events=[('queue',0,1,'Queue'),('prefill',1,3,'Prefill → logits'),('sample',3,4,'Sample t1'),('pass2',5,6,'Generate t2'),('pass3',6,7,'Generate t3')]
 ends=[0]+[e[2] for e in events[:3]]+[5,6,7,8]
 objects=[obj('service','lanes','Service execution','compute',{'lanes':['Service'],'start':0,'end':8,'axisLabel':'Declared schematic event units','ticks':[0,1,3,4,5,6,7,8]},0,span=2),seq('buffer','Service output buffer',3,1),seq('client','Client received tokens',3,1,1,role='output')]
 steps=[]
 for j,t in enumerate(ends):
  segments=[dict(id=i,lane=0,start=a,end=b,label=l,state='done' if b<=t else 'waiting',role='compute') for i,a,b,l in events]
  produced=sum(j>=b for b in [3,5,6])
  buf=[('t'+str(k+1)) if k<produced else None for k in range(3)]
  # Delivery consumes the service transport buffer, not generation history.
  cli=[None]*3
  if j>=4:cli[0]='t1';buf[0]=None
  if j==len(ends)-1:cli=['t'+str(k+1) for k in range(3)];buf=[None]*3
  titles=['Request enters the service queue','Finish queueing','Prefill produces initial logits','Sample the first token','Deliver the first token','Generate the second token','Generate the third token','Deliver the buffered later tokens']
  exps=['No output has been sampled or delivered.','The request can now begin prompt processing.','Prefill reads the prompt and produces the logits used for the first sample.','Sampling turns the initial logits into t1; the client is still waiting.','The first arrival ends client TTFT after queueing, prefill, sampling, and delivery.','One subsequent generation pass produces t2; transport retains it.','A second subsequent pass produces t3; neither later token has arrived yet.','Transport delivers t2 and t3 in one chunk; arrivals conceal the separate generation passes.']
  later=max(0,produced-1)
  metrics=[metric('prefill','Server prefill span',events[1][2]-events[1][1],'schematic units'),metric('ttft','Client TTFT boundary',ends[4]-ends[0],'schematic units'),metric('produced','Sampled tokens',produced,'tokens'),metric('received','Client tokens',nonnull(cli),'tokens'),metric('later','Subsequent generation passes',later,'passes')]
  # Stable token overlays preserve identity across transport delivery; cells remain the accounting baseline.
  tokens=[dict(id='t'+str(k+1),label='t'+str(k+1),value='t'+str(k+1),role='output',owner='Client' if cli[k] is not None else 'Service transport buffer',at=dict(object='client' if cli[k] is not None else 'buffer',index=k),state='resident') for k in range(produced)]
  steps.append(step('event'+str(j),titles[j],exps[j],{'service':{'segments':segments,'cursor':t,'status':'Service boundaries are visible'},'buffer':S(buf,'Unsent output tokens'),'client':S(cli,'Client arrivals')},metrics,['The client is waiting before the service reads the prompt.','Queueing has ended; prompt processing has not.','Prefill has finished, but the client has no token yet.','The first token exists at the service, while client TTFT continues.','The first delivery ends client TTFT; later generation is still ahead.','The second token remains buffered instead of arriving immediately.','Distinct generation passes have produced both buffered later tokens.','One client arrival carries tokens from separate service passes.'][j],None if j==0 else 'execute' if j in [1,2,3,5,6] else 'move',['service'] if j in [1,2,3,5,6] else ['t1'] if j==4 else ['t2','t3'] if j==7 else ['buffer','client'],tokens=tokens))
 return lesson('fig-vol3-first-token-accounting','First token: service work versus client arrival','Which event ends prefill, and which event ends client TTFT?','Three output tokens on a declared schematic event axis. Coordinates are illustrative, not measured latency. Transport buffering is shown explicitly.','schedule',objects,steps,['Prefill supplies the logits for the first token; later output tokens each require a subsequent pass.','Sampling and client receipt are distinct events.','A chunk can carry multiple tokens without merging their service generation passes.'])

def cancellation():
 objects=[seq('prefix','Received fragments: retained, never dispatched',4,0,labels=['open','key','bad key','later']),reg('parser','Local parser state',0,1),pool('kv','Service request KV allocation',4,1),seq('signal','Cancellation signal',1,1,1),seq('dispatch','Tool dispatch',1,2,role='output')]
 def case(late=False):
  steps=[];frags=['{','"query":"x",','"force":','...'];prefix=[None]*4;resident=['KV'+str(i+1) for i in range(2)]+[None]*2
  states={'prefix':S(prefix),'parser':R('Pending'),'kv':P(resident,'Request A'),'signal':S([None]),'dispatch':S([None])}
  actions=[('wait','Stream starts','The runtime holds every partial proposal until validation completes.',None),('open','Receive an open object','The prefix is incomplete but still completable; no tool call is authorized.','write'),('key','Receive a permitted key','A permitted key still does not make the partial proposal dispatchable.','write'),('reject','Reject an unknown key','The received force key is absent from this declared schema. The local gate rejects the proposal.','write'),('cancel','Send the cancellation request','Sending a signal does not prove that the remote generation loop stopped.','write'),('honor' if not late else 'late','Honor cancellation' if not late else 'Finish an already-running service step','Acknowledgment stops future generation before KV release.' if not late else 'A late cancellation permits one more service step; the local rejection remains in force.','execute'),('release','Release service KV state','The request has no remaining service consumer; its private state can be freed.','release')]
  for j,(i,title,exp,kind) in enumerate(actions):
   if j in [1,2,3]:prefix[j-1]=frags[j-1]
   if late and j==5:prefix[3]=frags[3];resident[2]='KV3'
   if j==6:resident=[None]*4
   states['prefix']=S(prefix,'Diagnostic prefix retained');states['parser']=R('Rejected' if j>=3 else 'Pending');states['kv']=P(resident,'Request A','No consumers remain' if j==6 else 'Service-owned');states['signal']=S(['acknowledged' if j>=5 and not late else 'sent' if j>=4 else None]);states['dispatch']=S([None],'Gate closed')
   metrics=[metric('held','Received fragments',nonnull(prefix),'fragments'),metric('occupied','Resident KV entries',nonnull(resident),'entries'),metric('dispatched','Dispatched proposals',0,'proposals'),metric('extra','Post-rejection service steps',int(late and j>=5),'steps')]
   subjects=['prefix','parser'] if j<=3 else ['signal'] if j==4 else ['kv','signal']
   steps.append(step(i,title,exp,states,metrics,['The local gate holds an incomplete proposal.','An open object remains completable and cannot be dispatched.','The permitted key preserves completable parser state, not dispatch permission.','The unknown key closes the dispatch gate before remote cancellation.','Remote decoding may still continue after the cancel signal leaves.','The local gate stays closed even though the service finished more work.' if late else 'Acknowledgment stops generation while service state remains allocated.','Service state is free; the rejected diagnostic prefix remains.'][j],kind,subjects))
  return steps
 return lesson('fig-vol3-streaming-cancellation','Reject locally, cancel remotely','What does rejection guarantee before a cancellation is honored?','Toy stream fragments and KV entries; the only permitted key is query. This lifecycle does not reproduce the chapter’s 4,096-token timing estimate.','ownership',objects,case(),['No incomplete or rejected proposal reaches tool dispatch.','Only the service can stop its decode loop and release its execution state.','The rejected diagnostic prefix remains available after service memory is released.'],checks=[{'kind':'metricEqualsOccupied','metric':'occupied','object':'kv'}],variants=[{'id':'late','label':'Late cancellation','scope':'Same malformed proposal; an already-running step finishes before service shutdown.','steps':case(True)}])

def context():
 old=['R0','R1','T17a','T17b','L0','L1'];new=old[:2]+['T18a']+old[3:]
 objects=[seq('old','Previous serialized tokens',len(old),0),seq('new','Next serialized tokens',len(old),0,1),pool('cache','KV working set during request selection',len(old),1),seq('boundary','First changed position',1,1,1)]
 def case(change):
  target=old.copy();target[change]='L2' if change==4 else 'T18a'
  common=next((i for i,(a,b) in enumerate(zip(old,target)) if a!=b),len(old));cache=['KV('+x+')' for x in old];steps=[]
  acts=[('retained','Previous request state is resident','The engine retains compatible state for the prior exact token prefix.',None),('edit','Change one serialized token','Only one token changes, but later attention states depend on the entire earlier prefix.','write'),('select','Select only the identical leading prefix','The old suffix may remain cached for the old request, but cannot serve this changed request.','release'),('prefill','Recompute the changed suffix','One bounded suffix prefill computes replacement KV for every position from the first mismatch onward.','execute')]
  for j,(i,title,exp,kind) in enumerate(acts):
   if j==2:cache=[v if k<common else None for k,v in enumerate(cache)]
   if j==3:cache=['KV('+x+')' for x in target]
   states={'old':S(old,'Reference request remains unchanged'),'new':S(old if j==0 else target,'Root / trunk / leaf tokens'),'cache':P(cache,'Request working set',['Old-request KV retained','Old suffix awaits compatibility selection','Only compatible prefix selected','Fresh suffix KV computed for changed request'][j]),'boundary':S([None if j==0 else common],'Zero-based token index')}
   metrics=[metric('reuse','Reusable leading positions',len(old) if j==0 else common,'positions'),metric('required','Suffix positions requiring prefill',0 if j==0 else len(old)-common,'positions'),metric('occupied','Resident working-set KV',nonnull(cache),'entries')]
   steps.append(step(i,title,exp,states,metrics,['All prior KV positions are resident for the old request.','The mismatch ends reusable prefix state even where later token text matches.','Only preceding KV is selected; the incompatible suffix is absent.','Fresh KV replaces the entire suffix, including unchanged token text.'][j],kind,['new'] if j==1 else ['cache','boundary']))
  return steps
 return lesson('fig-vol3-context-zones','First mismatch ends prefix reuse','Why must unchanged suffix tokens be processed again?','Six illustrative serialized token positions: two root, two trunk, two leaf. Matching model, position, attention configuration, permissions, and residency are assumed. Old-request KV remains conceptually distinct from next-request selection.','cache',objects,case(2),['The previous token reference never changes.','Next-request reuse ends at the first differing serialized token.','Removing old suffix entries from this selection does not assert global cache eviction.'],checks=[{'kind':'constantValues','object':'old'},{'kind':'metricEqualsOccupied','metric':'occupied','object':'cache'}],variants=[{'id':'leaf','label':'Only the leaf changes','scope':'Change token position four instead of the trunk; the longer preceding prefix remains compatible.','steps':case(4)}])

def paged():
 block=4;length=10;table=[3,7,9];private=14;pos=9;logical=pos//block;offset=pos%block;tail=length%block
 labels=[str(i) for i in range(block)]
 objects=[seq('A','Candidate A block table',len(table),0),seq('B','Candidate B block table',len(table),0,1),seq('shared','Physical block 9: original tail',block,1,role='memory',labels=labels),seq('private','Physical block 14: private tail',block,1,1,role='memory',labels=labels),reg('lookup','Read address for logical position 9',2),reg('owners','Physical tail ownership',2,1)]
 states={'A':S(table),'B':S([None]*len(table)),'shared':S(list(range(length-tail,length))+[None]*(block-tail),'A-owned initialized KV'),'private':S([None]*block,'No allocated block yet'),'lookup':R(None),'owners':R('9: A; 14: free')};steps=[]
 actions=[('initial','Candidate A retains the original tail','Full blocks 3 and 7 and the initialized tail at block 9 belong to A.',None),('resolve','Resolve logical position 9','Integer division selects the table entry; modulo selects the initialized slot.','write'),('fork','Fork B by sharing the prefix','Both candidates reference the existing blocks; no KV entry has been copied.','write'),('allocate','Reserve private block 14 for B','The private tail is allocated before any initialized entry is copied.','write'),('copy','Copy initialized tail entries','Copy only positions 8 and 9. Unwritten capacity remains null; the source tail survives.','copy'),('remap','Remap only B’s tail table entry','B now resolves its tail to the initialized private copy; A still resolves to block 9.','write'),('append','Append position 10 to B’s private tail','The append initializes one free entry without modifying A’s original tail.','write')]
 allocated=False;copies=0
 for j,(i,title,exp,kind) in enumerate(actions):
  if j==1:states['lookup']=R(f'logical {logical} → block {table[logical]}, slot {offset}')
  if j==2:states['B']=S(table);states['owners']=R('9: A + B; 14: free')
  if j==3:allocated=True;states['private']['status']='Allocated to B; all entries unwritten';states['owners']=R('9: A + B; 14: B')
  if j==4:states['private']=S(list(range(length-tail,length))+[None]*(block-tail),'B-owned initialized KV');copies=tail
  if j==5:states['B']=S(table[:-1]+[private]);states['owners']=R('9: A; 14: B')
  if j==6:states['private']=S(list(range(length-tail,length+1))+[None]*(block-tail-1),'B-owned initialized KV')
  # Allocation is separate from valid KV residency: a register value records ownership.
  if j>=3:states['lookup']=R(f'Private block {private}: allocated to B; source slot {offset}')
  tok=[]
  if j==4:
   tok=[dict(id='tail-copy'+str(k),label=str(length-tail+k),value=length-tail+k,role='memory',owner='Candidate B',at={'object':'private','index':k},state='copy',sourceAt={'object':'shared','index':k}) for k in range(tail)]
  used=sum(x is not None for x in states['private']['values'])
  metrics=[metric('allocated','Private tail blocks',int(allocated),'blocks'),metric('copied','Copied initialized entries',copies,'entries'),metric('privateused','Initialized private slots',used,'slots'),metric('slack','Private unwritten capacity',block-used if allocated else 0,'slots')]
  steps.append(step(i,title,exp,states,metrics,['The original tail has valid entries followed by unwritten capacity.','The selected table entry locates the block; the remainder locates its slot.','Both tables reference one original tail without duplicating its initialized KV.','B owns an empty private block; no copied KV is readable there yet.','The private tail now contains the initialized source entries; the source remains intact.','Only B changes its mapping, leaving A’s original tail reachable.','B’s appended KV occupies its private tail while A’s sequence stays unchanged.'][j],kind,['shared','private'] if j==4 else ['A','B','lookup','private','owners'],fr={'object':'shared'} if j==4 else None,to={'object':'private'} if j==4 else None,tokens=tok))
 return lesson('fig-vol3-pagedattention-mapping','Share a prefix, privatize a tail','Which KV entries must be copied before a fork appends?','The source’s ten-token prefix, four-token blocks, tables [3,7,9], and private block 14. Full blocks 3 and 7 are summarized by their table entries; only tails are expanded. Tail sequences distinguish unwritten null entries from valid KV; the ownership register records whether the physical block is allocated.','ownership',objects,steps,['Candidate A’s table and original initialized entries survive B’s append.','Copy-on-write copies initialized tail entries only.','B’s table changes after its private copy is initialized, before its append.'],checks=[{'kind':'constantValues','object':'A'}])

def radix():
 # Compressed nodes each own one schematic exclusive KV block.
 nodes=['Root','Repair','Test','Rejected','Other','Final'];parents=[None,0,1,1,0,4];active={0,1,2}
 objects=[obj('tree','plot','Prefix ancestry and eviction frontier','memory',{'x':{'min':0,'max':4,'label':'Branch placement','ticks':[]},'y':{'min':0,'max':3,'label':'Prefix depth (schematic)','ticks':[]}},0,span=2,narrow=0),pool('blocks','Exclusively held KV blocks',len(nodes),1),reg('freed','Outstanding allocator demand',1,1,typ='Integer',unit='blocks')]
 xy=[(2,3),(1,2),(0,1),(2,1),(3,2),(4,1)];steps=[]
 # Evictions are one leaf each; pressure recorded via register to create real state.
 retained=set(range(len(nodes)));freed=[]
 actions=[('resident','Active paths coexist with dormant cache','Root, Repair, and Test have an active request. Other remains internal while Final is retained.',None),('pressure','Request free blocks','The allocator requests three blocks; dormant internal nodes remain protected by their descendants.','write'),('final','Evict dormant leaf Final','Final has no child and no execution reference. Remove its index entry and return its exclusive block.','release'),('other','Evict newly exposed leaf Other','Other becomes a leaf only after Final disappears; it now has no retained descendant.','release'),('rejected','Evict the independent dormant leaf Rejected','Rejected can be removed without touching the live Root–Repair–Test path.','release')]
 for j,(i,title,exp,kind) in enumerate(actions):
  if j>=2:
   n=[5,4,3][j-2];retained.remove(n);freed.append(nodes[n])
  children=lambda n:[k for k,p in enumerate(parents) if p==n and k in retained]
  eligible=[n for n in retained if n not in active and not children(n)]
  series=[dict(id='edge'+str(n),label=nodes[p]+' → '+nodes[n],role='memory',points=[list(xy[p]),list(xy[n])]) for n,p in enumerate(parents) if p is not None and n in retained]
  markers=[dict(id='node'+str(n),x=xy[n][0],y=xy[n][1],label=nodes[n]+(' · active' if n in active else ' · eligible leaf' if n in eligible else ' · retained parent')) for n in sorted(retained)]
  slots=[n if k in retained else None for k,n in enumerate(nodes)]
  states={'tree':{'series':series,'markers':markers,'status':'Edges require their ancestors'},'blocks':P(slots,'Prefix cache'),'freed':R(0 if j==0 else len([5,4,3])-len(freed))}
  metrics=[metric('occupied','Retained exclusive blocks',nonnull(slots),'blocks'),metric('eligible','Eligible dormant leaves',len(eligible),'nodes'),metric('released','Returned blocks',len(freed),'blocks')]
  steps.append(step(i,title,exp,states,metrics,['Dormant Other cannot be evicted while its Final child remains.','Allocator demand does not make a retained ancestor safe to evict.','Removing Final exposes Other as a newly eligible leaf.','Other can now disappear without breaking any retained path.','Only the active request path remains; dormant exclusive blocks are free.'][j],kind,['tree','blocks','freed']))
 return lesson('fig-vol3-radix-tree-cache','Evict leaves before their ancestors','Why can a dormant parent remain non-evictable?','Six compressed radix nodes, one exclusively held schematic block per node. The active request pins Root–Repair–Test. No outstanding device operations or concurrent lookups remain on evicted leaves; synchronization is assumed.','cache',objects,steps,['No retained node loses its required ancestor.','Active request blocks are never evicted.','Returning an exclusive block removes its physical pool occupancy.'],checks=[{'kind':'metricEqualsOccupied','metric':'occupied','object':'blocks'},{'kind':'nondecreasingMetric','metric':'released'}])

def chunked():
 # Reduced pedagogical schedule: same workload and declared work units in both cases.
 chunks=4;prompt_units=8;decode_units=1;total=prompt_units+chunks*decode_units
 objects=[obj('schedule','lanes','One shared worker: prefill versus active decode','compute',{'lanes':['Long prompt','Active decode'],'start':0,'end':total,'axisLabel':'Declared schematic work units','ticks':list(range(0,total+1,2))},0,span=2,narrow=0),seq('prompt','Prompt chunks completed',chunks,1),seq('output','Active stream output tokens',chunks,1,1,role='output')]
 def case(mono):
  segments=[]
  if mono:
   segments=[dict(id='prefill',lane=0,start=0,end=prompt_units,label='Entire long prompt',role='compute')]
   segments += [dict(id='d'+str(k),lane=1,start=prompt_units+k*decode_units,end=prompt_units+(k+1)*decode_units,label='Decode '+str(k+1),role='compute') for k in range(chunks)]
  else:
   t=0
   for k in range(chunks):
    segments.extend([dict(id='p'+str(k),lane=0,start=t,end=t+prompt_units//chunks,label='Chunk '+str(k+1),role='compute'),dict(id='d'+str(k),lane=1,start=t+prompt_units//chunks,end=t+prompt_units//chunks+decode_units,label='Decode '+str(k+1),role='compute')]);t=segments[-1]['end']
  # At most nine snapshots: initialize plus one per bounded task.
  steps=[];completed=set()
  for j in range(len(segments)+1):
   if j:completed.add(segments[j-1]['id'])
   now=0 if j==0 else segments[j-1]['end'];pdone=chunks if mono and 'prefill' in completed else sum('p'+str(k) in completed for k in range(chunks));ddone=sum('d'+str(k) in completed for k in range(chunks))
   states={'schedule':{'segments':[dict(s,state='done' if s['id'] in completed else 'waiting') for s in segments],'cursor':now,'status':'Tasks execute serially on one worker'},'prompt':S(['P'+str(k+1) if k<pdone else None for k in range(chunks)]),'output':S(['t'+str(k+1) if k<ddone else None for k in range(chunks)])}
   arrivals=[s['end'] for s in segments if s['lane']==1];gaps=[b-a for a,b in zip([0]+arrivals,arrivals)]
   metrics=[metric('prefilled','Completed prompt chunks',pdone,'chunks'),metric('tokens','Active-stream tokens',ddone,'tokens'),metric('first','First active-stream arrival',arrivals[0],'schematic units'),metric('promptend','Prompt completion boundary',max(s['end'] for s in segments if s['lane']==0),'schematic units'),metric('maxgap','Maximum active-stream gap',max(gaps),'schematic units')]
   title='Same queued workload' if j==0 else 'Finish '+segments[j-1]['label'].lower()
   exp='One long prompt shares a worker with one active decode stream.' if j==0 else 'This bounded task finishes before the next scheduled task begins.'
   steps.append(step('task'+str(j),title,exp,states,metrics,('Neither request has progressed beyond its initial boundary.' if j==0 else ('The prompt has completed before the active stream can resume.' if mono else 'The prompt advances while the active stream awaits its next admitted task.') if segments[j-1]['lane']==0 else 'The active stream advances while unfinished prompt chunks remain.' if pdone<chunks else 'The active stream advances after the long prompt has completed.'),None if j==0 else 'execute',['schedule','prompt','output']))
  return steps
 return lesson('fig-vol3-chunked-timeline','Chunking exposes scheduling boundaries','What progress can an active decode make between prefill chunks?','Reduced scheduling model: four equal prompt chunks, one ongoing decode stream, eight units of total prefill work, one unit per decode, serial execution and no overlap. This is not the chapter’s 4,096-token/32-decode weight-only timing model; attention and launch overhead are omitted.','schedule',objects,case(False),['Both schedules process identical prompt work and emit the same active-stream tokens.','One worker executes one task at a time.','Chunking changes admission boundaries and prompt completion, not the amount of declared prefill work.'],variants=[{'id':'monolithic','label':'Monolithic prefill','scope':'Same work and serial-worker assumptions; admit the entire prompt before active decodes.','steps':case(True)}])

def search():
 objects=[pool('candidates','Generated candidate frontier',6,0),seq('selected','Retained next-stage prefixes',2,0,1,role='output'),seq('path','Selected tree path',3,1),reg('visits','Retained root → B edge visits',1,1,typ='Integer',unit='visits'),reg('mean','Retained root → B edge mean',2,typ='Scalar',unit='score')]
 def case(mode):
  scores=[2,7,4,9];names=['a','b','c','d'];slots=[None]*6;sel=[None]*2;path=[None]*3;n=0;q=None;steps=[]
  if mode=='flat':actions=[('initial','Independent calls are ready',None),('generate','Complete four independent candidates','execute'),('score','Score completed candidates','write'),('select','Keep the best completed candidate','release')]
  elif mode=='beam':actions=[('initial','One prompt starts the beam',None),('expand','Generate the first two extensions','execute'),('retain','Retain both initial prefixes','write'),('expandagain','Expand each survivor into two children','execute'),('score','Score the expanded frontier','write'),('prune','Keep only the top two children','release')]
  else:actions=[('initial','The retained root → B edge has one prior visit',None),('selectpath','Select one root-to-leaf path','write'),('expand','Generate one new leaf','execute'),('evaluate','Evaluate that new leaf','write'),('backup','Update counts and means along the path','execute')]
  for j,(i,title,kind) in enumerate(actions):
   if mode=='flat':
    if j==1:slots=names+[None]*2
    if j==2:slots=[f'{a} · score {v}' for a,v in zip(names,scores)]+[None]*2
    if j==3:
     best=max(range(len(scores)),key=scores.__getitem__);sel[0]=names[best];slots=[f'{a} · score {v}' if k==best else None for k,(a,v) in enumerate(zip(names,scores))]+[None]*2
    exp=['Independent candidates have no cross-candidate generation dependency.','The independent calls finish before the selector runs.','The final scorer attaches a declared value to each complete candidate.','Final selection removes losing complete candidates; their full generation work has already occurred.'][j]
   elif mode=='beam':
    if j==1:slots=['a','b']+[None]*4
    if j==2:sel=['a','b'];slots=['a','b']+[None]*4
    if j==3:slots=['a1','a2','b1','b2']+[None]*2;sel=[None]*2
    if j==4:slots=[f'{a} · score {v}' for a,v in zip(['a1','a2','b1','b2'],scores)]+[None]*2
    if j==5:
     keep=sorted(range(len(scores)),key=scores.__getitem__,reverse=True)[:2];sel=[['a1','a2','b1','b2'][k] for k in keep];slots=[v if k in keep else None for k,v in enumerate(slots)]
    exp=['The beam starts as one prompt, not two complete candidates.','The prompt yields two initial prefixes in one expansion.','Both survive this first boundary; retained width is two.','Each retained prefix yields two children in parallel; peak frontier now exceeds retained beam width.','Each new partial candidate receives a declared step score.','Only two children survive for further generation; the others lose their live candidate state.'][j]
   else:
    history=[2];value=8
    n=len(history);q=sum(history)/len(history)
    if j>=1:path=['root','branch B','new leaf']
    if j>=2:slots[0]='B1 · unscored'
    if j>=3:slots[0]=f'B1 · value {value}'
    if j>=4:n=len(history+[value]);q=sum(history+[value])/n
    exp=['The retained root → B edge has one earlier leaf value; the next leaf is unexpanded.','The selection policy chooses one retained branch; other branches need not expand now.','One new leaf becomes available under the selected branch.','A leaf scorer returns the declared toy value.','The retained root → B edge adds this visit and folds the leaf value into its running mean. The new leaf edge has separate statistics, omitted here.'][j]
   states={'candidates':P(slots,'Search frontier'),'selected':S(sel),'path':S(path,'One selected ancestry path'),'visits':R(n,'Previously retained root → B edge only' if mode=='tree' else 'Tree bookkeeping unused'),'mean':R(q,'Retained root → B edge mean' if mode=='tree' else 'Tree bookkeeping unused')}
   generated=(len(names) if mode=='flat' and j>=1 else 2+(4 if j>=3 else 0) if mode=='beam' and j>=1 else int(mode=='tree' and j>=2))
   metrics=[metric('live','Live generated candidates',nonnull(slots),'candidates'),metric('retained','Next-stage survivors',nonnull(sel),'candidates'),metric('generated','Generated candidates',generated,'candidates'),metric('visits','Retained-edge visits',n,'visits')]
   steps.append(step(i,title,exp,states,metrics,exp,kind,['candidates','selected','path','visits','mean']))
  return steps
 return lesson('fig-vol3-search-topologies','Where search retains and prunes state','At which boundary does generated state stop consuming the frontier?','Declared candidate labels and toy scorer values, without any accuracy claim. Cases are distinct search procedures, not equal-budget performance comparisons. Tree case displays backup statistics only for the previously retained root → B edge; the newly created leaf edge has its own statistics, omitted here. Full PUCT selection is not modeled.','ownership',objects,case('beam'),['Generated candidates and retained next-stage survivors are distinct counts.','Beam width limits survivors, not the expanded frontier.','Displayed tree statistics belong only to the previously retained root → B edge, never the newly created leaf edge.'],checks=[{'kind':'metricEqualsOccupied','metric':'live','object':'candidates'}],variants=[{'id':'flat','label':'Best of four','scope':'Complete independent candidates before final selection. Toy scores match the beam scorer values but are applied to complete candidates.','steps':case('flat')},{'id':'tree','label':'One tree-search visit','scope':'The previously retained root → B edge starts with prior leaf value 2 and receives one newly evaluated value 8. Display its visit and mean update; omit separate new-leaf-edge statistics.','steps':case('tree')}])

def access():
 # Explicit undirected toy graph; query distance is one-dimensional for clarity.
 nodes={'A':(0,1),'M':(2,1),'B':(4,1),'C':(5,2)};edges=[('A','M'),('M','B'),('B','C')];allowed={'A','B','C'};query=(5,1)
 plotconf={'x':{'min':0,'max':6,'label':'Toy embedding coordinate','ticks':list(range(7))},'y':{'min':0,'max':3,'label':'Schematic second coordinate','ticks':[]}}
 objects=[obj('graph','plot','Traversal graph after scope checks','memory',plotconf,0,span=2,narrow=0),seq('walk','Visited eligible nodes',3,1),seq('result','Retrieved eligible nodes',3,1,1,role='output'),reg('frontier','Walk frontier',2)]
 def reachable(start,es):
  seen={start};stack=[start]
  while stack:
   a=stack.pop()
   for u,v in es:
    b=v if u==a else u if v==a else None
    if b is not None and b not in seen:seen.add(b);stack.append(b)
  return seen
 def case(exact=False):
  steps=[]
  acts=[('unfiltered','Inspect the full graph',None),('mask','Apply eligibility before traversal','transform'),('visit','Visit eligible entry A','write'),('fallback','Score every eligible node' if exact else 'Seed a second entry at B','execute' if exact else 'write'),('retrieve','Return the nearest visited eligible node','write')]
  for j,(i,title,kind) in enumerate(acts):
   es=edges if j==0 else [(u,v) for u,v in edges if u in allowed and v in allowed]
   visited=[] if j<2 else ['A'];front='Not started' if j<2 else 'Empty: A has no eligible neighbor'
   if j>=3:visited=sorted(allowed) if exact else ['A','B'];front='Exact eligible scan complete' if exact else 'B seeds the formerly unreachable region'
   out=[None]*3
   if j==4:
    best=min(visited,key=lambda n:sum((a-b)**2 for a,b in zip(nodes[n],query)));out[0]=best
   markers=[dict(id=n,x=x,y=y,label=n+(' · masked' if j>=1 and n not in allowed else ' · visited' if n in visited else '')) for n,(x,y) in nodes.items()]
   markers.append(dict(id='query',x=query[0],y=query[1],label='q'))
   series=[dict(id=u+v,label=u+' ↔ '+v,role='memory',points=[list(nodes[u]),list(nodes[v])]) for u,v in es]
   states={'graph':{'series':series,'markers':markers,'status':'Masked nodes remain visible as a reference but have no traversal links'},'walk':S(visited+[None]*(3-len(visited))),'result':S(out),'frontier':R(front)}
   reach=reachable('A',es)&allowed
   metrics=[metric('eligible','Eligible graph nodes',len(allowed),'nodes'),metric('reachable','Eligible nodes reachable from A',len(reach),'nodes'),metric('visited','Eligible nodes visited',len(visited),'nodes'),metric('returned','Eligible nodes returned',nonnull(out),'nodes')]
   exps=['A connects to the near-query region only through masked node M.','Scope checks remove both links incident on M. Eligible B and C remain but disconnect from A.','A is visited; no eligible outgoing edge can advance the walk. The frontier is now empty.','The fallback scores all eligible vectors without requiring graph connectivity.' if exact else 'A second eligible entry bypasses the disconnected bridge; the first walk still cannot cross M.','Rank only visited eligible nodes under the declared toy Euclidean distance.']
   steps.append(step(i,title,exps[j],states,metrics,['The bridge through M is the only route from A toward the query.','Removing M’s traversal links disconnects eligible B and C from A.','The walk stalls at A despite eligible nodes near the query.','Every eligible node becomes a scored candidate without using traversal links.' if exact else 'A new entry reaches B without traversing forbidden M.','The selected result belongs to the eligible visited set.'][j],kind,['graph','walk','result','frontier']))
  return steps
 return lesson('fig-vol3-access-prefilter','Scope can disconnect a graph walk','How can eligible neighbors remain unreachable from a valid entry?','Four-node undirected toy graph with eligibility mask {A,B,C}; M is forbidden. Coordinates illustrate graph connectivity and a declared Euclidean ranking example, not measured ANN recall. Masked node markers are reference only; removed links cannot be traversed.','spatial',objects,case(),['Traversal never visits masked M.','Masking removes traversal edges, not the existence of eligible remote nodes.','Extra entries and exact scanning add eligible candidates without relaxing the scope mask.'],variants=[{'id':'exact','label':'Exact eligible fallback','scope':'Same graph, mask, and query; score all eligible nodes instead of adding an entry point.','steps':case(True)}])

def build():return [first_token(),cancellation(),search(),context(),chunked(),paged(),radix(),access()]
def main():
 expected=build();path=pathlib.Path(__file__).with_name('vol3-a.json')
 if '--write' in sys.argv:
  path.write_text(json.dumps(expected,ensure_ascii=False,indent=2)+'\n');print('Wrote',path);return
 actual=json.loads(path.read_text());assert actual==expected,'Authored snapshots differ from recomputed declared models'
 for l in actual:
  for case in [{'id':'base','steps':l['steps']}]+l.get('variants',[]):
   end=case['steps'][-1]
   print(l['key'],case['id'],end['id'],json.dumps(end['snapshot']['metrics'],ensure_ascii=False))
 print('All snapshots, derived metrics, indices, candidate scores, interval boundaries, occupancy and endpoint states match recomputation.')
if __name__=='__main__':main()

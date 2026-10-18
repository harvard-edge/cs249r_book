#!/usr/bin/env python3
"""Recompute scoped mechanism examples; default verifies JSON without changing it."""
import json, math, sys
from pathlib import Path
from copy import deepcopy


def obj(id,kind,label,role,config,row=0,col=0,span=1):
 return dict(id=id,kind=kind,label=label,role=role,config=config,layout=dict(wide=dict(row=row,column=col,columnSpan=span),narrow=dict(row=row*2+col,column=0,columnSpan=1)))
def seq(id,label,n,row=0,col=0,role='input',labels=None):
 c=dict(slots=n)
 if labels:c['slotLabels']=labels
 return obj(id,'sequence',label,role,c,row,col)
def pool(id,label,n,row=0,col=0,labels=None):
 c=dict(slots=n,columns=min(n,4))
 if labels:c['slotLabels']=labels
 return obj(id,'pool',label,'memory',c,row,col)
def reg(id,label,row=0,col=0,unit='schematic units'):
 return obj(id,'register',label,'compute',dict(type='SCALAR',unit=unit),row,col)
def mat(id,label,r,c,row=0,col=0,role='memory',rl=None,cl=None):
 conf=dict(rows=r,columns=c)
 if rl:conf['rowLabels']=rl
 if cl:conf['columnLabels']=cl
 return obj(id,'matrix',label,role,conf,row,col)
def ss(values,status=''):return dict(values=values,status=status)
def ps(slots,status=''):return dict(slots=slots,status=status)
def rs(value,status='',**kw):return dict(value=value,status=status,**kw)
def slot(v,owner,status='retained'):return dict(value=v,owner=owner,status=status)
def metric(id,label,value,unit):return dict(id=id,label=label,value=value,unit=unit)
def trans(kind,subjects,meaning,fr=None,to=None):
 d=dict(kind=kind,subjects=subjects,meaning=meaning)
 if fr:d['from']=dict(object=fr)
 if to:d['to']=dict(object=to)
 return d
def token(id,label,value,owner,at,index,role='input',state='resident',sourceAt=None):
 d=dict(id=id,label=label,value=value,role=role,owner=owner,at=dict(object=at,index=index),state=state)
 if sourceAt:d['sourceAt']=dict(object=sourceAt[0],index=sourceAt[1])
 return d
def step(id,title,explanation,objects,metrics,result,tr=None,tokens=None):
 return dict(id=id,title=title,explanation=explanation,snapshot=dict(objects=deepcopy(objects),tokens=deepcopy(tokens or []),metrics=deepcopy(metrics),activeConnections=[],result=result),transition=tr)
def lesson(key,title,question,scope,family,objects,invariants):
 return dict(schema=1,key='vol2:'+key,title=title,question=question,scope=scope,family=family,objects=objects,steps=[],invariants=invariants)

def fsdp():
 l=lesson('fig-fsdp-parameter-lifetime','Release gathered weights, retain saved activations','Why can gathered weights disappear while activations must remain?',
 'One rank and one FSDP unit; reshard after forward, no prefetch. Slots count logical allocations, not bytes; shard storage may alias full buffers. Optimizer update is outside this scene.',
 'ownership',[pool('persistent','Persistent shards',2,0,0,['Parameters','Optimizer']),pool('temporary','Temporary unit state',3,0,1,['Full parameters','Activations','Gradient'])],
 ['Parameter and optimizer shards persist throughout this unit.','Forward activations survive parameter resharding until backward consumes them.','Gathered weights are required again before backward; logical slot counts do not imply byte totals.'])
 p=[slot('P shard','rank 0'),slot('O shard','rank 0')]; tmp=[None]*3
 events=[('start','Shards retained','The rank begins with its persistent ownership.',None),('gather-f','Gather forward parameters','AllGather reconstructs the full unit on this rank.','write'),('forward','Execute forward','Forward creates saved activations for backward.','execute'),('reshard-f','Release forward gathered parameters','The forward consumer has finished; saved activations still have a future consumer.','release'),('gather-b','Gather backward parameters','Backward needs full parameters again under this resharding policy.','write'),('backward','Execute backward','Backward consumes the saved activations and produces a full local gradient.','execute'),('release-a','Release consumed activations','The final activation consumer has completed.','release'),('release-p','Release backward gathered parameters','Backward no longer needs the gathered parameter buffer.','release'),('reduce','ReduceScatter the full gradient','The full local gradient becomes this rank’s reduced gradient shard.','transform')]
 for i,(id,title,ex,k) in enumerate(events):
  if id in ('gather-f','gather-b'):tmp[0]=slot('P full','rank 0')
  if id=='forward':tmp[1]=slot('A saved','rank 0')
  if id=='reshard-f':tmp[0]=None
  if id=='backward':tmp[2]=slot('G full','rank 0')
  if id=='release-a':tmp[1]=None
  if id=='reduce':tmp[2]=slot('G shard','rank 0')
  if id=='release-p':tmp[0]=None
  occupied=sum(s is not None for s in tmp)
  st=dict(persistent=ps(p,'Remain allocated'),temporary=ps(tmp,'Last-consumer lifetimes'))
  m=[metric('temporary','Live temporary allocations',occupied,'logical allocations'),metric('persistent','Persistent shards',len(p),'logical allocations')]
  result={'start':'Persistent shards establish the rank’s allocation floor before full weights arrive.','gather-f':'Full weights now permit forward computation; the parameter shard remains owned.','forward':'Backward now has a future claim on the saved activations.','reshard-f':'Full weights are gone, but saved activations cannot be released before backward.','gather-b':'Backward has both its reconstructed weights and its saved forward activations.','backward':'The gradient is available; backward has completed its last reads of activations and gathered weights.','release-a':'Consumed activations no longer occupy memory; the gradient remains available for reduction.','release-p':'Gathered weights are gone; the full gradient still awaits reduction.','reduce':'ReduceScatter leaves a gradient shard alongside the persistent parameter and optimizer shards.'}[id]
  subjects=['temporary']; tr=None if k is None else trans(k,subjects,ex)
  l['steps'].append(step(id,title,ex,st,m,result,tr))
 l['checks']=[dict(kind='metricEqualsOccupied',metric='temporary',object='temporary'),dict(kind='metricEqualsOccupied',metric='persistent',object='persistent')]
 return l

def moe():
 # Declared top-1 routes; scalar expert functions stand in for FFNs.
 homes=[0,0,1,1]; routes=[1,0,1,1]; inputs=[1,2,3,4]; scales=[2,3]; gates=[1,.5,.25,1]
 values=[inputs[i]*scales[routes[i]] for i in range(len(inputs))]; combined=[values[i]*gates[i] for i in range(len(inputs))]
 l=lesson('fig-moe-all-to-all-routing','Dispatch, compute, and return each token','How does an expert result get back to the rank that owns its token?',
 'Toy top-1 forward pass: two ranks, one expert per rank, four scalar tokens a–d. E0(x)=2x and E1(x)=3x stand in for FFNs; declared gate weights are [1,0.5,0.25,1]. No capacity drops or backward pass.',
 'communication',[pool('home','Source-rank token slots',4,0,0,['R0 a','R0 b','R1 c','R1 d']),pool('experts','Expert queues',4,0,1,['E0','E1','E1','E1']),seq('return','Returned expert results',4,1,0,'output',['a','b','c','d']),seq('output','Gate-weighted source output',4,1,1,'output',['a','b','c','d'])],
 ['Each token has one selected expert in this top-1 example.','Dispatch and return preserve token identity and source-rank identity.','Only selected experts compute each token; a returned result still requires the declared gate weight.'])
 order=sorted(range(len(inputs)),key=lambda i:(routes[i],i)); positions={idx:pos for pos,idx in enumerate(order)}
 for phase in range(5):
  hs=[slot(f'{chr(97+i)}={inputs[i]}',f'rank {homes[i]}') for i in range(len(inputs))] if phase==0 else [None]*len(inputs)
  es=[None]*len(inputs)
  if phase in (1,2):
   for i in range(len(inputs)):es[positions[i]]=slot(f'{chr(97+i)}={inputs[i] if phase==1 else values[i]}',f'rank {routes[i]} / E{routes[i]}','ready')
  ret=[None]*len(inputs) if phase<3 else values
  out=[None]*len(inputs) if phase<4 else combined
  toks=[]
  if phase in (0,1,2,3):
   for i in range(len(inputs)):
    at='home' if phase==0 else 'experts' if phase in (1,2) else 'return';idx=positions[i] if at=='experts' else i
    toks.append(token(f't{i}',chr(97+i),f'{chr(97+i)}={inputs[i] if phase<2 else values[i]}',f'rank {homes[i]}' if at!='experts' else f'rank {routes[i]} / E{routes[i]}',at,idx))
  loads=[routes.count(e) for e in range(len(scales))]
  m=[metric('dispatched','Dispatched tokens',len(inputs) if phase>=1 else 0,'tokens'),metric('maxload','Largest selected expert queue',max(loads) if phase>=1 else 0,'tokens'),metric('returned','Results returned',sum(v is not None for v in ret),'tokens')]
  names=['Source payloads ready','Dispatch one AllToAll','Execute selected expert functions','Return one AllToAll','Apply the source gate weights'];ids=['start','dispatch','expert','return','combine']
  ex=['Token identity includes its home rank and selected expert.','Payloads transfer to expert owners; the input values do not change.','Expert computation changes payload values while preserving a–d identity.','Each expert result moves back to its own source slot.','Source ranks multiply returned results by their declared top-1 gate weights.'][phase]
  kinds=[None,'move','execute','move','execute']; tr=None if phase==0 else trans(kinds[phase],[f't{i}' for i in range(len(inputs))] if phase<4 else ['output'],ex)
  l['steps'].append(step(ids[phase],names[phase],ex,dict(home=ps(hs,'Source ownership'),experts=ps(es,'E0 receives b; E1 receives a, c, d'),**{'return':ss(ret,'Results at source ranks'),'output':ss(out,'Gate weights applied only at source')}),m,['The selected routes determine each token’s next owner before any expert computes.','E1 receives a, c, and d while E0 receives b; sparse routing can produce unequal queues.','Expert results have changed, but their token identities and home ranks still determine the return route.','The source ranks have their expert results back; no gate weighting has yet been applied.','Applying the declared gate weights completes each source rank’s top-1 output.'][phase],tr,toks))
 return l

def pipeline():
 stages=2; batches=2; duration=1; slots=2*(batches+stages-1)
 tasks=[]
 for s in range(stages):
  for mb in range(batches):
   fstart=mb+s; bstart=batches+stages-1+(stages-1-s)+(batches-1-mb)
   for direction,start in [('F',fstart),('B',bstart)]:tasks.append(dict(id=f'{direction}{s}{mb}',lane=s,start=start,end=start+duration,label=f'{direction} m{mb+1}',state='waiting',role='compute'))
 l=lesson('fig-pipeline-parallelism','Follow GPipe dependencies and saved activations','When can a stage release a microbatch’s saved activations?',
 'Toy balanced GPipe: two stages, two microbatches; all forward tasks precede backward, in reverse microbatch order. One slot per task, instantaneous stage communication, no optimizer update or measured time.',
 'schedule',[obj('schedule','lanes','Fixed GPipe schedule','compute',dict(lanes=['Stage 0','Stage 1'],start=0,end=slots,axisLabel='Schematic task slots',ticks=list(range(slots+1))),0,0,2),pool('saved','Saved activations by stage and microbatch',stages*batches,1,0,['S0 m1','S0 m2','S1 m1','S1 m2']),reg('done','Completed stage tasks',1,1,'tasks')],
 ['A stage runs forward only after upstream forward and backward only after downstream backward.','Every backward task consumes its matching stage/microbatch activation.','Task slots describe a balanced toy schedule; idle capacity is not a measured time prediction.'])
 for t in range(slots+1):
  segments=[dict(task,state='done' if task['end']<=t else 'waiting') for task in tasks]
  saved=[]
  for s in range(stages):
   for mb in range(batches):
    f=next(z for z in tasks if z['id']==f'F{s}{mb}');b=next(z for z in tasks if z['id']==f'B{s}{mb}')
    saved.append(slot(f'A m{mb+1}',f'stage {s}') if f['end']<=t<b['end'] else None)
  done=sum(z['state']=='done' for z in segments);idle=stages*t-done
  st=dict(schedule=dict(segments=segments,cursor=t,status='One boundary per action'),saved=ps(saved,'Released only after matching backward'),done=rs(done,'Stage tasks finished'))
  m=[metric('saved','Live saved activations',sum(v is not None for v in saved),'activation records'),metric('idle','Elapsed idle stage-slots',idle,'stage-slots')]
  ex='No stage has executed yet.' if t==0 else 'Complete the tasks occupying this slot in parallel; dependencies determine which stages can work.'
  l['steps'].append(step(f'slot{t}','Before the first task' if t==0 else f'Complete schedule slot {t}',ex,st,m,['Stage 1 must wait for stage 0 to produce the first microbatch’s activation.','Stage 0 can admit m2 while stage 1 consumes m1’s forward activation.','Stage 0 has finished forward work but cannot run backward until a downstream gradient arrives.','All forward work is complete; stage 1 can begin backward with m2.','Stage 1’s m2 gradient now permits stage 0’s m2 backward; stage 1 can also backpropagate m1.','Stage 0 still needs its m1 backward; stage 1 has finished both microbatches.','Every saved activation has reached its final backward consumer and is released.'][t],None if t==0 else trans('execute',['schedule','saved','done'],ex)))
 l['checks']=[dict(kind='metricEqualsOccupied',metric='saved',object='saved')]
 return l

def ulysses():
 n=4; rows=[f't{i}' for i in range(n)];heads=[f'h{i}' for i in range(n)]
 l=lesson('fig-ulysses-ownership-transpose','Transpose ownership, compute heads, transpose back','Why does a rank need every token for only its assigned heads?',
 'Four ranks, four tokens, four ordinary attention heads; head dimension omitted. Cells name token/head identity and owning rank. The local attention step marks outputs, not numerical attention values. No GQA or KV replication.',
 'ownership',[mat('reference','Original token ownership',n,n,0,0,'reference',rows,heads),mat('working','Current tile ownership',n,n,0,1,'memory',rows,heads),seq('headstate','Local head outputs',n,1,0,'compute',heads)],
 ['AllToAll changes ownership of token/head tiles without deleting token–head pairs.','Each rank receives every token for its assigned head before local attention.','The inverse transpose restores token ownership; output values differ from input activations.'])
 orig=[f'R{r}' for r in range(n) for c in range(n)]
 for phase in range(4):
  owners=orig if phase in (0,3) else [f'R{c}' for r in range(n) for c in range(n)]
  vals=[f'{"O" if phase>=2 else "X"} / {owners[r*n+c]}' for r in range(n) for c in range(n)]
  st=dict(reference=ss(orig,'Ownership reference; not an extra tensor allocation'),working=ss(vals,'Rows are tokens; columns are heads'),headstate=ss([f'O h{c}' if phase>=2 else None for c in range(n)],'Each completed head consumed every token'))
  moved=sum(r!=c for r in range(n) for c in range(n)) if phase>=1 else 0
  m=[metric('tiles','Token–head tiles',n*n,'tiles'),metric('moved','Remote tiles per completed exchange',moved,'tiles'),metric('heads','Completed local heads',n if phase>=2 else 0,'heads')]
  ex=['Input sequence shards assign each token row to one rank.','AllToAll assigns each head column to one rank, exposing a complete sequence per head.','Every rank computes its complete head over all tokens; X becomes output O.','The inverse AllToAll sends each output tile back to its token-owning rank.'][phase]
  l['steps'].append(step(['start','transpose','attention','inverse'][phase],['Start with token rows','Transpose to head columns','Execute complete-head attention','Restore token-row ownership'][phase],ex,st,m,['Each rank owns one token row across every head.','Each rank now owns a whole head column, so local attention can see that head’s full sequence.','Complete-head attention has produced output O without exchanging data between heads.','Subsequent layers receive token-owned output rows again; attention has changed values while the inverse transpose restores ownership.'][phase],None if phase==0 else trans(['','move','execute','move'][phase],['working','headstate'] if phase==2 else ['working'],ex)))
 l['checks']=[dict(kind='constantValues',object='reference')]
 return l

def checkpoint():
 checkpoint_step=4; later_step=7
 l=lesson('fig-checkpoint-recovery-timeline','Failure rewinds live progress to committed progress','Which progress survives a failure during a synchronous checkpoint save?',
 'Toy synchronous checkpoint: committed step 4, compute through step 7, then fail during a new save. Save-in-flight never becomes durable; loading and restart are distinct actions. Steps count training progress, not elapsed seconds.',
 'ownership',[reg('live','Live training progress',0,0,'training steps'),reg('durable','Committed restore point',0,1,'training steps'),pool('save','Save in flight',1,1,0,['Pending generation']),seq('rework','Lost training steps',later_step-checkpoint_step,1,1,'output'),reg('process','Process execution state',2,0,'state')],
 ['Only a completed, committed checkpoint is a restore point.','A synchronous save does not advance training steps.','Failure loses progress beyond the committed step; recovery latency is separate from repeated training.'])
 live=checkpoint_step; durable=checkpoint_step; save=None; lost=[None]*(later_step-checkpoint_step)
 events=[('start','Committed checkpoint available','Both live and durable progress begin at the same committed step.',''),('compute','Advance training to step 7','Training produces live progress beyond the durable restore point.','execute'),('save','Start a synchronous save','Training pauses while the new generation is written; the old checkpoint remains committed.','copy'),('fail','Fail before save completion','The failed process and its incomplete save are lost; committed progress survives.','release'),('load','Load the committed checkpoint','Read the last completed generation into new live process state.','copy'),('restart','Complete restart and warmup','The restored process becomes ready without advancing training progress.','execute'),('repeat','Repeat the lost training steps','Productive compute returns live progress to the pre-failure point.','execute')]
 ready=False;restart=0
 for id,title,ex,k in events:
  if id=='compute':live=later_step
  if id=='save':save=slot(live,'checkpoint writer','inflight')
  if id=='fail':lost=list(range(durable+1,live+1));live=None;save=None
  if id=='load':live=durable
  if id=='restart':restart+=1;ready=True
  if id=='repeat':live=later_step
  st=dict(live=rs(live,'Ready' if ready else 'Unavailable' if live is None else 'Training state'),durable=rs(durable,'Immutable committed generation'),save=ps([save],'Incomplete generations are not restore points'),rework=ss(lost,'Must be recomputed after restart'),process=rs({'start':'running','compute':'running','save':'saving','fail':'failed','load':'restored','restart':'ready','repeat':'running'}[id],'Recovery does not advance training'))
  toks=[]
  if id=='save':
   toks=[token('save-copy','Pending step',live,'checkpoint writer','save',0,'memory','copy',('live',0))];toks[0]['sourceAt'].pop('index')
  if id=='load':
   toks=[token('restore-copy','Restore',durable,'training process','live',0,'memory','copy',('durable',0))];toks[0]['sourceAt'].pop('index');toks[0]['at'].pop('index')
  m=[metric('lost','Lost progress',sum(x is not None for x in lost),'training steps'),metric('pending','Uncommitted saves',int(save is not None),'generations'),metric('restart','Completed restarts',restart,'restarts')]
  tr=None if not k else trans(k,['save'] if id=='save' else ['process'] if id=='restart' else ['live'],ex)
  l['steps'].append(step(id,title,ex,st,m,{'start':'The committed checkpoint can restore the current live state.','compute':'The extra live training steps are exposed to loss until another checkpoint commits.','save':'An incomplete save cannot replace the prior committed restore point.','fail':'The incomplete generation is lost; recovery must use the old committed checkpoint.','load':'The new process has old committed progress, so the lost steps remain to be repeated.','restart':'Scheduling and warmup have finished; recovery has not recreated the lost training work.','repeat':'Recomputation restores the lost live progress, while the committed checkpoint remains unchanged.'}[id],tr,toks))
 return l

def gang():
 devices=8; requirement=devices;half=devices//2
 l=lesson('fig-gang-scheduling-deadlock','Allocated GPUs can still make no progress','How does atomic admission prevent this hold-and-wait cycle?',
 'Eight GPU slots; rigid jobs J1 and J2 each require all eight before executing. No preemption or timeouts. Compare partial admission with atomic gang admission; execution is one schematic job completion, not measured duration.',
 'ownership',[pool('gpus','GPU ownership',devices,0,0,[f'GPU {i}' for i in range(devices)]),seq('queue','Waiting jobs',2,0,1,'control'),reg('completed','Completed jobs',1,0,'jobs')],
 ['A job executes only when it owns its entire GPU requirement.','Waiting for admission does not confer ownership under the gang policy.','Gang admission prevents this partial-hold cycle; it does not eliminate every scheduling deadlock or fragmentation source.'])
 def case(atomic):
  entries=['start','arrival','grant-one','execute-one','release-one','grant-two','execute-two','release-two'] if atomic else ['start','arrival','grant-one','grant-two']
  holders=[None]*devices;waiting=['J1',None];completed=[];steps=[]
  for id in entries:
   k='';ex='J1 waits for admission; GPUs have no owners.'
   if id=='arrival':waiting[1]='J2';k='reveal';ex='J2 arrives while J1 is still waiting.'
   if id=='grant-one':
    for i in range(devices if atomic else half):holders[i]=slot('J1','J1','ready' if atomic else 'allocated')
    if atomic:waiting[0]=None
    k='write';ex='Allocate J1’s entire requirement atomically.' if atomic else 'Allocate only half of J1’s requirement; J1 retains the lease while waiting.'
   if id=='grant-two':
    for i in range(devices) if atomic else range(half,devices):holders[i]=slot('J2','J2','ready' if atomic else 'allocated')
    if atomic:waiting[1]=None
    k='write';ex='After J1 releases its gang, allocate all GPUs to J2.' if atomic else 'Give the remaining GPUs to J2; both jobs now hold partial gangs and wait.'
   if id.startswith('execute'):
    completed.append('J1' if id=='execute-one' else 'J2');k='execute';ex='The admitted job finishes its computation while retaining its GPU lease.'
   if id.startswith('release'):
    holders=[None]*devices;k='release';ex='Release the completed job’s entire GPU lease.'
   counts={j:sum(s is not None and s['owner']==j for s in holders) for j in ['J1','J2']}
   executable=sum(c==requirement and j not in completed for j,c in counts.items())
   st=dict(gpus=ps(holders,'A partial gang cannot execute'),queue=ss(waiting,'Waiters do not imply free GPUs'),completed=rs(len(completed),'Finished jobs'))
   m=[metric('held','GPUs held',sum(s is not None for s in holders),'GPUs'),metric('eligible','Jobs with a full gang',executable,'jobs'),metric('completed','Completed jobs',len(completed),'jobs')]
   title={'start':'J1 waits for admission','arrival':'J2 joins the waiting queue','grant-one':'Allocate GPUs to J1','grant-two':'Allocate GPUs to J2','execute-one':'Execute J1','execute-two':'Execute J2','release-one':'Release J1’s lease','release-two':'Release J2’s lease'}[id]
   result='Both jobs hold GPUs, but neither has a complete gang.' if not atomic and id=='grant-two' else 'Only a complete admitted gang can make useful progress.'
   subjects=['queue'] if id=='arrival' else ['completed'] if id.startswith('execute') else ['gpus']
   steps.append(step(id,title,ex,st,m,result,None if not k else trans(k,subjects,ex)))
  return steps
 l['steps']=case(False);l['variants']=[dict(id='atomic',label='Atomic gang admission',scope='Same jobs and GPU pool; grant only a full gang, keeping the other job unallocated until release.',steps=case(True))]
 l['checks']=[dict(kind='metricEqualsOccupied',metric='held',object='gpus')]
 return l

def fusion():
 x=[-1,2];weight=2; linear=[weight*v for v in x];relu=[max(0,v) for v in linear];normal=[v/max(relu) for v in relu]; n=len(x)
 l=lesson('fig-fusion-before-after','Keep equal intermediates on chip','Which HBM transfers disappear when compatible operators fuse?',
 'Toy vectors x=[−1,2], linear=2x, ReLU, then normalization by the vector maximum. Compare separate kernels with a feasible fused kernel. Count intermediate element transfers only; common input, weights, final output, metadata and time are excluded.',
 'spatial',[seq('input','Common input reference',n,0,0,'reference'),seq('local','Registers / SRAM working vector',n,0,1,'compute'),pool('hbm','HBM intermediate cells',n*2,1,0,['I1[0]','I1[1]','I2[0]','I2[1]']),seq('output','Final normalized output',n,1,1,'output')],
 ['Both cases perform the same three arithmetic operations and produce the same final vector.','The counted traffic includes only writes and reads of the two intermediate vectors.','The fused case assumes compatible operations and sufficient registers/SRAM; traffic reduction does not specify latency reduction.'])
 def case(fused):
  events=['start','linear','relu','normalize','final'] if fused else ['start','linear','write-one','read-one','relu','write-two','read-two','normalize','final']
  local=[None]*n;hbm=[None]*(n*2);out=[None]*n;traffic=0;steps=[]
  for id in events:
   k=''; toks=[];ex='Input is a reference shared by both cases; the working vector has not been produced.'
   if id=='linear':local=linear.copy();k='execute';ex='Multiply the input by the declared scalar weight.'
   if id=='relu':local=relu.copy();k='execute';ex='Apply ReLU to the resident working vector.'
   if id=='normalize':local=normal.copy();k='execute';ex='Normalize by the maximum using the same arithmetic in either case.'
   if id.startswith('write'):
    offset=0 if id=='write-one' else n
    for j,v in enumerate(local):hbm[offset+j]=slot(v,'I1' if offset==0 else 'I2')
    local=[None]*n;traffic+=n;k='write';ex='The separate kernel writes its intermediate to HBM and ends its local lifetime.'
   if id.startswith('read'):
    offset=0 if id=='read-one' else n;local=[hbm[offset+j]['value'] for j in range(n)];traffic+=n;k='copy';ex='The next kernel reads the retained HBM intermediate without erasing it.'
    toks=[token(f'read{offset+j}',f'I{1+offset//n}[{j}]',local[j],'next kernel','local',j,'memory','copy',('hbm',offset+j)) for j in range(n)]
   if id=='final':out=local.copy();k='write';ex='Expose the final result; its common output write is outside the intermediate-traffic count.'
   st=dict(input=ss(x,'Reference, common to both schedules'),local=ss(local,'Fused residency' if fused else 'One kernel’s working vector'),hbm=ps(hbm,'Retained intermediate copies' if not fused else 'No intermediate allocation'),output=ss(out,'Same numerical output'))
   m=[metric('traffic','Intermediate HBM transfers',traffic,'elements'),metric('resident','Resident HBM intermediate cells',sum(v is not None for v in hbm),'elements')]
   titles={'start':'Before the chain executes','linear':'Compute the linear vector','relu':'Compute the activation vector','normalize':'Compute the normalized vector','final':'Write the final output','write-one':'Write intermediate I1 to HBM','read-one':'Read intermediate I1 from HBM','write-two':'Write intermediate I2 to HBM','read-two':'Read intermediate I2 from HBM'}
   subjects=['hbm'] if id.startswith('write') else [t['id'] for t in toks] if toks else ['output'] if id=='final' else ['local']
   steps.append(step(id,titles[id],ex,st,m,{'start':'The two cases begin with the same input and the same arithmetic.','linear':'The linear intermediate is resident on chip and ready for its next consumer.','write-one':'The next separate kernel must reread I1 because its producer’s local lifetime has ended.','read-one':'The activation kernel can use its copy while I1 remains in HBM.','relu':'The activation intermediate remains on chip.' if fused else 'The activation intermediate must cross the next separate-kernel boundary.','write-two':'The normalization kernel must reread I2 because its producer’s local lifetime has ended.','read-two':'Normalization can use its copy while I2 remains in HBM.','normalize':'Normalization has produced the same final vector using the same arithmetic.','final':'The fused chain avoids both intermediate HBM round-trips.' if fused else 'Separate kernels required both intermediate HBM round-trips; the final vector agrees with fusion.'}[id],None if not k else trans(k,subjects,ex),toks))
  return steps
 l['steps']=case(False);l['variants']=[dict(id='fused',label='Feasible fused kernel',scope='Same toy arithmetic; both intermediate vectors remain in registers/SRAM until consumed. Common final output is still written.',steps=case(True))]
 l['checks']=[dict(kind='constantValues',object='input'),dict(kind='metricEqualsOccupied',metric='resident',object='hbm'),dict(kind='nondecreasingMetric',metric='traffic')]
 return l

def flash():
 q=1.;keys=[0.,math.log(2)];values=[2.,4.]; scores=[q*k for k in keys]; running=[];m=-math.inf;den=0.;num=0.
 for score,v in zip(scores,values):
  newm=max(m,score);alpha=0. if m==-math.inf else math.exp(m-newm);p=math.exp(score-newm);den=alpha*den+p;num=alpha*num+p*v;m=newm;running.append((m,den,num))
 exact=sum(math.exp(s)*v for s,v in zip(scores,values))/sum(math.exp(s) for s in scores)
 assert math.isclose(num/den,exact)
 rnd=lambda x:round(x,6)
 l=lesson('fig-performance-engineering-flashattention-tiling','Rescale retained state as each attention tile arrives','How can a later larger score update exact softmax without retaining all scores?',
 'One scalar query, one head, two one-key tiles, unmasked toy attention with no score scaling: q=1, k=[0,ln 2], v=[2,4]. Six-decimal display; calculations use full precision. The whole two-key example fits on chip but deliberately uses one-key tiles. No backward or hardware traffic estimate.',
 'reduction',[seq('hbm','HBM operands and final output',6,0,0,'memory',['q','k0','v0','k1','v1','O']),obj('tile','chamber','On-chip q / k / v / score','compute',dict(slots=4),0,1),reg('maximum','Running maximum m',1,0),reg('normalizer','Running normalization l',1,1),reg('accumulator','Unnormalized output a',2,0),reg('output','On-chip normalized output',2,1)],
 ['Every relevant key/value tile contributes exactly once for this query.','When m increases, both the old normalization sum and old output accumulator rescale by exp(old m − new m).','HBM operands remain unchanged by reads; score tiles never become an HBM score matrix.','The final a/l equals direct softmax attention, up to display rounding.'])
 hbm=[q,rnd(keys[0]),values[0],rnd(keys[1]),values[1],None];tile=[None]*4;m=None;den=0.;num=0.;out=None;loaded=0
 events=['start','load0','score0','update0','load1','score1','update1','normalize','write']
 for id in events:
  k='';ex='The query and both key/value tiles are available in HBM.';toks=[]
  if id.startswith('load'):
   j=int(id[-1]);tile=[q,rnd(keys[j]),values[j],None];loaded+=1;k='copy';ex='Read the next key/value tile into reusable on-chip space; retain the query.'
   for name,source,dest,val in [('k',1+j*2,1,tile[1]),('v',2+j*2,2,tile[2])]:toks.append(token(f'{name}{j}-read',name+str(j),val,'query tile','tile',dest,'input','copy',('hbm',source)))
   if j==0:toks.append(token('q-read','q',q,'query tile','tile',0,'input','copy',('hbm',0)))
  if id.startswith('score'):
   j=int(id[-1]);tile[3]=rnd(scores[j]);k='execute';ex='Compute this tile’s q·k score locally; HBM has no score cell.'
  if id.startswith('update'):
   j=int(id[-1]);m,den,num=running[j];k='execute';ex='Update m, l and a using the current score and value.' if j==0 else 'The new maximum is ln 2: rescale old l and a by exp(−ln 2), then add the current tile.'
  if id=='normalize':out=num/den;k='execute';ex='After the final required tile, divide the retained accumulator by its normalization sum.'
  if id=='write':hbm[-1]=rnd(out);k='write';ex='Write the final query output to HBM; operands stay resident and unchanged.'
  st=dict(hbm=ss(hbm,'Operands persist; only O is newly written'),tile=ss(tile,'Four stable on-chip slots'),maximum=rs(None if m is None else rnd(m),'Maximum over processed scores'),normalizer=rs(rnd(den),'Rescaled sum of exponentials'),accumulator=rs(rnd(num),'Rescaled weighted-value sum'),output=rs(None if out is None else rnd(out),'Normalized only after all tiles'))
  processed=0 if id in ('start','load0','score0') else 1 if id in ('update0','load1','score1') else len(keys)
  metrics=[metric('tiles','K/V tiles loaded',loaded,'tiles'),metric('processed','Tiles accumulated',processed,'tiles'),metric('scores','Resident HBM score cells',0,'cells')]
  titles={'start':'Before tile reads','load0':'Load query and first K/V tile','score0':'Compute the first score','update0':'Accumulate the first tile','load1':'Load the second K/V tile','score1':'Compute the second score','update1':'Rescale and accumulate the second tile','normalize':'Normalize the retained accumulator','write':'Write the final output to HBM'}
  subjects=[t['id'] for t in toks] if toks else ['tile'] if id.startswith('score') else ['maximum','normalizer','accumulator'] if id.startswith('update') else ['output'] if id=='normalize' else ['hbm']
  l['steps'].append(step(id,titles[id],ex,st,metrics,{'start':'No score matrix or attention output has been allocated in HBM.','load0':'The first tile can compute on chip while its source operands remain readable in HBM.','score0':'The score is temporary on-chip state; it is not an HBM matrix entry.','update0':'The running normalization and output accumulator now represent the first key/value contribution.','load1':'The previous tile’s contribution survives in m, l, and a even as its operand slots are reused.','score1':'The larger second score requires changing the reference maximum before contributions can be combined.','update1':'Rescaling both old l and old a preserves the first tile’s contribution under the new maximum.','normalize':'Dividing a by l gives the same attention output as direct softmax over both scores.','write':'Only the final output is written; the temporary score tiles never became a resident HBM matrix.'}[id],None if not k else trans(k,subjects,ex),toks))
 return l

def batching():
 lengths={'R1':1,'R2':4,'R3':2,'R4':3,'R5':2}; initial=['R1','R2','R3','R4'];capacity=len(initial)
 l=lesson('fig-continuous-batching-comparison','Release cache, then admit at an iteration boundary','When does a finished slot become useful to a waiting request?',
 'Prefix of a toy four-slot trace: R1–R4 need [1,4,2,3] decode steps, and R5 waits with two decode steps after prefill. Show the first completion and next decode only. One abstract KV lease per request; capacity permits replacement. Prefill and decode work units are counted separately from elapsed time.',
 'schedule',[seq('members','Current iteration membership',capacity,0,0,'compute'),pool('kv','Request-owned KV leases',capacity,0,1,[f'Slot {i+1}' for i in range(capacity)]),seq('waiting','Waiting arrival',1,1,0,'control'),mat('trace','Work actually executed',capacity,3,1,1,'output',[f'Slot {i+1}' for i in range(capacity)],['Decode 1','Boundary','Decode 2'])],
 ['A finished response and a reusable batch slot are separate scheduling decisions.','Replacement requires both a waiting request and sufficient KV capacity.','Prefill is an actual operation before R5 can decode; slot-work counts do not establish throughput.','All original request IDs, lengths, and initial membership agree between policies.'])
 def case(continuous):
  events=['start','decode-one','release','prefill','decode-two'] if continuous else ['start','decode-one','release','decode-two']
  members=initial.copy();kv=[slot('KV '+r,r) for r in initial];progress={r:0 for r in lengths};trace=[None]*(capacity*3);waiting=['R5'];work=0;steps=[]
  for id in events:
   ex='The same initial requests own four slots and KV leases; R5 waits without a lease.';k=''
   if id.startswith('decode'):
    column=0 if id=='decode-one' else 2
    for i,r in enumerate(members):
     if r is not None and progress[r]<lengths[r]:
      progress[r]+=1;trace[i*3+column]=r+(' final' if progress[r]==lengths[r] else ' decode');work+=1
    k='execute';ex='Each active request executes one decode step in parallel; completed requests do no further decode work.'
   if id=='release':members[0]=None;kv[0]=None;k='release';ex='R1 is complete. Release its KV lease and its active slot before considering replacement.'
   if id=='prefill':members[0]='R5';kv[0]=slot('KV R5','R5');waiting=[None];trace[1]='R5 prefill';work+=1;k='execute';ex='At the next iteration boundary, admit waiting R5 and run its prefill within the free capacity.'
   labels=[None if r is None else f'{r} {progress[r]}/{lengths[r]}' for r in members]
   st=dict(members=ss(labels,'Completed requests remain visible until released'),kv=ps(kv,'Abstract leases, not equal physical KV bytes'),waiting=ss(waiting,'Admission depends on cache capacity'),trace=ss(trace,'Blank cells performed no work'))
   metrics=[metric('resident','Live KV leases',sum(s is not None for s in kv),'leases'),metric('work','Productive slot operations',work,'slot-operations'),metric('waiting','Waiting requests',sum(r is not None for r in waiting),'requests')]
   titles={'start':'Initial requests admitted','decode-one':'Complete the first decode iteration','release':'Release completed R1','prefill':'Prefill waiting R5 into the free slot','decode-two':'Complete the next decode iteration'}
   result='Static membership leaves the released slot unused until the batch ends.' if not continuous and id in ('release','decode-two') else 'R5 can contribute only after release, admission, and prefill.'
   steps.append(step(id,titles[id],ex,st,metrics,result,None if not k else trans(k,['kv','members'] if id=='release' else ['trace','members'],ex)))
  return steps
 l['steps']=case(False);l['variants']=[dict(id='continuous',label='Continuous admission',scope='Same initial requests, lengths, and arrival; the released slot admits R5 at the boundary with an explicit prefill operation.',steps=case(True))]
 l['checks']=[dict(kind='metricEqualsOccupied',metric='resident',object='kv'),dict(kind='metricEqualsNonNull',metric='work',object='trace'),dict(kind='nondecreasingMetric',metric='work')]
 return l

def build():return [fsdp(),moe(),pipeline(),ulysses(),checkpoint(),gang(),fusion(),flash(),batching()]

def main():
 path=Path(__file__).with_name('vol2-b.json');expected=build()
 if '--write' in sys.argv:
  path.write_text(json.dumps(expected,ensure_ascii=False,indent=2)+'\n')
 else:
  actual=json.loads(path.read_text());assert actual==expected,'Authored JSON differs from recomputed complete snapshots'
 for l in expected:
  for name,steps in [('base',l['steps'])]+[(v['id'],v['steps']) for v in l.get('variants',[])]:
   for current in steps:
    print(l['key'],name,current['id'],json.dumps(current['snapshot']['metrics'],ensure_ascii=False))
 print('Verified complete snapshots from declared toy inputs; no hardware specification registry.')
if __name__=='__main__':main()

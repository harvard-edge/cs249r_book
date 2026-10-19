#!/usr/bin/env python3
import json,copy,pathlib
L=[]
def obj(id,kind,label,role,config,n):
 return dict(id=id,kind=kind,label=label,role=role,config=config,layout={'wide':{'row':n//2,'column':n%2},'narrow':{'row':n,'column':0}})
def mat(id,label,role,r,c,n,**kw):return obj(id,'matrix',label,role,dict(rows=r,columns=c,**kw),n)
def seq(id,label,role,s,n):return obj(id,'sequence',label,role,{'slots':s},n)
def pool(id,label,s,n):return obj(id,'pool',label,'memory',{'slots':s,'columns':min(s,4)},n)
def reg(id,label,n):return obj(id,'register',label,'control',{'type':'Schematic scalar'},n)
def state(values,status=''):return {'values':values,'status':status}
def slot(value,owner,status='retained'):return dict(value=value,owner=owner,status=status)
def ps(slots,status=''):return dict(slots=slots,status=status)
def metric(id,label,value,unit):return dict(id=id,label=label,value=value,unit=unit)
def step(id,title,ex,objects,metrics,result,kind=None,subjects=None,tokens=None,**kw):
 return dict(id=id,title=title,explanation=ex,snapshot=dict(objects=copy.deepcopy(objects),tokens=tokens or [],metrics=metrics,activeConnections=[],result=result),transition=None if kind is None else dict(kind=kind,subjects=subjects or [],meaning=ex,**kw))
def lesson(key,title,q,scope,family,objects,steps,invariants,checks=None,**kw):
 x=dict(schema=1,key='vol1:'+key,title=title,question=q,scope=scope,family=family,objects=objects,steps=steps,invariants=invariants,**kw)
 if checks:x['checks']=checks
 L.append(x);return x
# Sector utilization, independent from address dependence.
objects=[seq('requests','Requested byte addresses','input',4,0),mat('words','Address space: one row per aligned sector','memory',4,4,1),pool('sectors','Fetched sectors for the four requests',4,2),reg('pointer','Separate pointer lookup: payload address',3),reg('payload','Separate dependent payload read',4)]
def sectorcase(addresses):
 sectors=list(dict.fromkeys(a//16 for a in addresses)); slots=[None]*4; words=list(range(0,64,4)); steps=[]
 def snap(id,title,ex,kind=None):
  count=sum(s is not None for s in slots);o={'requests':state(addresses),'words':state(words),'sectors':ps(slots),'pointer':dict(value=ptr,status='Independent dependency example'),'payload':dict(value=val,status='Payload waits for resolved address')}
  steps.append(step(id,title,ex,o,[metric('fetched','Fetched for request set',count*16,'B'),metric('useful','Useful request payload',len(addresses)*4,'B')],('The payload becomes readable only after its address is known.' if id=='read' else 'Address 48 is now available; the dependent read may issue next.' if id=='resolve' else 'Fetched sectors retain source words; unused bytes still count as traffic.' if id.startswith('fetch') else 'Adjacent words can share a sector; separated words may require distinct sectors.'),kind,['sectors'] if id.startswith('fetch') else ['pointer' if id=='resolve' else 'payload']))
 ptr=val=None;snap('ready','Establish the cold request set','Each request reads one four-byte word. No reuse or prefetch is modeled.')
 for i,s in enumerate(sectors):
  slots[i]=slot(f'{s*16}–{(s+1)*16-1}','Request set');snap('fetch'+str(i),'Fetch one aligned sector','The interface fetches the entire sector and merges requests inside it.','write')
 ptr=48;snap('resolve','Resolve the separate payload address','An index read returns byte address 48. The payload read could not issue earlier.','reveal')
 val='payload@48';snap('read','Read the dependent payload','The resolved address now permits the payload request, regardless of unused bandwidth.','reveal')
 return steps
lesson('fig-irregular-address-transactions','Sector traffic and dependent addresses','Why can equal useful payload require different sector traffic?', 'Illustrative 4-byte words and aligned 16-byte sectors; cold cache, merged requests, no reuse. Separate pointer example excludes its traffic from request-set metrics.','spatial',objects,sectorcase([0,4,8,12]),['Sector fetches do not consume or alter source words.','Pointer resolution must precede the dependent payload request.'],[{'kind':'constantValues','object':'words'}],variants=[dict(id='scattered',label='Scattered addresses',scope='Same useful payload; one requested word in each of four sectors. Pointer example is unchanged.',steps=sectorcase([0,16,32,48]))])
# Source-faithful weight-stationary wave; each clock is one parallel action.
A=[[1,2],[3,4]];B=[[5,6],[7,8]];flat=lambda m:sum(m,[])
objects=[mat('a','Activation matrix A','input',2,2,0),mat('b','Stationary weights B: rows are reduction indices','memory',2,2,1),mat('arrivals','Activation at each PE this cycle','input',2,2,2),mat('partial','PE result this cycle; top results flow downward','compute',2,2,3),mat('c','Completed output C','output',2,2,4)]
steps=[];outputs=[None]*4
for t in range(-1,4):
 arrivals=[None]*4;sums=[None]*4;active=0
 for k in range(2):
  for j in range(2):
   i=t-k-j
   if 0<=i<2:
    active+=1;arrivals[k*2+j]=A[i][k];sums[k*2+j]=sum(A[i][kk]*B[kk][j] for kk in range(k+1))
    if k==1:outputs[i*2+j]=sums[k*2+j]
 o={'a':state(flat(A)),'b':state(flat(B),'Weights stay in their original PE registers'),'arrivals':state(arrivals,'Top row k=0; bottom row k=1'),'partial':state(sums,'Null is an idle PE this cycle'),'c':state(outputs)}
 steps.append(step('ready' if t<0 else 'cycle'+str(t),'Load the weight-stationary mapping' if t<0 else f'Execute conceptual cycle {t}', 'Each active PE multiplies one activation by its resident weight and adds the arriving top-row partial sum when k=1.' if t>=0 else 'A and B are declared toy integers. Lower-row activations enter one cycle later so reduction indices align.',o,[metric('active','Active PEs',active,'PEs'),metric('written','Completed outputs',sum(v is not None for v in outputs),'elements')], 'Partial sums advance down columns while activations advance right.' if t<3 else 'The final output drains after the leading PEs become idle.',None if t<0 else 'execute',['arrivals','partial','c'],tokens=[dict(id='activation'+str(t-k-j)+str(k),label='a'+str(t-k-j)+str(k),value=A[t-k-j][k],role='input',owner='Toy GEMM',at={'object':'arrivals','index':k*2+j},state='copy' if j==0 else 'resident',**({'sourceAt':{'object':'a','index':(t-k-j)*2+k}} if j==0 else {})) for k in range(2) for j in range(2) if 0<=t-k-j<2]))
lesson('fig-systolic-array','Aligned arrivals in a weight-stationary array','Why must lower-row activations arrive one cycle later?', 'Declared 2×2 integer product; B is stationary. One conceptual cycle includes parallel MAC and neighbor forwarding. UI motion is not hardware timing.','spatial',objects,steps,['PE row k holds B[k,j]; each contribution uses the matching reduction index.','A and stationary B remain unchanged.','An output appears only after both reduction terms have contributed.'],[{'kind':'constantValues','object':'a'},{'kind':'constantValues','object':'b'},{'kind':'metricEqualsNonNull','metric':'written','object':'c'}])
# Tiling, two reduction slices.
objects=[mat('a','A in backing memory','input',2,2,0),mat('b','B in backing memory','input',2,2,1),mat('localA','Local A slice','memory',2,1,2),mat('localB','Local B slice','memory',1,2,3),mat('partial','Retained local C tile','compute',2,2,4),mat('output','Written C in backing memory','output',2,2,5)]
steps=[];la=[None]*2;lb=[None]*2;c=[0]*4;out=[None]*4;reads=0
specs=[('ready','Establish a local C tile',None),('load0','Load reduction slice k=0','write'),('mac0','Accumulate slice k=0','execute'),('load1','Replace operands with slice k=1','write'),('mac1','Accumulate slice k=1 into retained C','execute'),('store','Write the completed C tile','write')]
for id,title,kind in specs:
 if id.startswith('load'):
  k=int(id[-1]);la=[A[i][k] for i in range(2)];lb=B[k][:];reads+=len(la)+len(lb)
 if id.startswith('mac'):
  c=[c[i*2+j]+la[i]*lb[j] for i in range(2) for j in range(2)]
 if id=='store':out=c[:]
 o=dict(a=state(flat(A)),b=state(flat(B)),localA=state(la),localB=state(lb),partial=state(c,'Retained across operand replacement'),output=state(out))
 ex='Only operand slices are replaced; the local C tile retains every earlier contribution.' if id=='load1' else 'One local outer product updates all four C accumulators.' if id.startswith('mac') else 'The full reduction must finish before the C tile is written.' if id=='store' else 'Two operand elements from each matrix fit beside the retained four-element C tile.'
 steps.append(step(id,title,ex,o,[metric('reads','Backing-memory operand reads',reads,'elements'),metric('written','Written output elements',sum(v is not None for v in out),'elements')], ('Backing memory now receives the complete C tile; earlier partial sums were never written out.' if id=='store' else 'Replacing operands preserves every contribution already accumulated in C.' if id=='load1' else 'Both reduction slices have contributed; the complete C tile can now be written.' if id=='mac1' else 'The first products remain local so the next slice can add to them.' if id=='mac0' else 'Resident operand slices can be reused across all four output accumulators.'),'copy' if id.startswith('load') else kind,['partial'] if id.startswith('mac') else ['output'] if id=='store' else ['localA','localB'],tokens=[] if id=='ready' else [dict(id='localA'+str(k)+str(i),label='A'+str(i)+str(k),value=la[i],role='input',owner='Toy GEMM',at={'object':'localA','index':i},state='copy' if id.startswith('load') else 'resident',**({'sourceAt':{'object':'a','index':i*2+k}} if id.startswith('load') else {})) for i in range(2)]+[dict(id='localB'+str(k)+str(j),label='B'+str(k)+str(j),value=lb[j],role='input',owner='Toy GEMM',at={'object':'localB','index':j},state='copy' if id.startswith('load') else 'resident',**({'sourceAt':{'object':'b','index':k*2+j}} if id.startswith('load') else {})) for j in range(2)]))
lesson('fig-tiling-diagram','Retain C across reduction tiles','What must remain resident when operand tiles are replaced?', 'Declared 2×2 integer GEMM with 2×1 A and 1×2 B slices; local capacity holds both operands and four C accumulators. Counts are elements, not a runtime prediction.','ownership',objects,steps,['Replacing A/B slices preserves the accumulated C tile.','Backing memory receives C only after the complete reduction.'],[{'kind':'constantValues','object':'a'},{'kind':'constantValues','object':'b'},{'kind':'metricEqualsNonNull','metric':'written','object':'output'}])
# Toy KV capacity; shape declares its own bytes, not a competing hardware registry.
objects=[pool('weights','Fixed toy weights',2,0),pool('cache','Per-request retained KV positions',8,1),seq('pending','Current consumed decode inputs','input',2,2),obj('growth','plot','Actual allocated KV footprint','output',{'x':{'min':0,'max':8,'label':'Retained token positions'},'y':{'min':0,'max':128,'label':'KV bytes'}},3)]
slots=[None]*8;steps=[];weights=[None]*2;inputs=[None,None]
for id in ['empty','weights','prefillA','prefillB','consume','append','complete','release']:
 if id=='weights':weights=[slot('W0','Model'),slot('W1','Model')]
 if id=='prefillA':slots[:2]=[slot('A:p0','A'),slot('A:p1','A')]
 if id=='prefillB':slots[2:4]=[slot('B:p0','B'),slot('B:p1','B')]
 if id=='consume':inputs=['A:t0','B:t0']
 if id=='append':slots[4:6]=[slot('A:t0','A'),slot('B:t0','B')]
 if id=='complete':inputs=['A:done','B:t0']
 if id=='release':slots=[None if s and s['owner']=='A' else s for s in slots]
 count=sum(s is not None for s in slots);bpt=2*2*1*2*2
 o={'weights':ps(weights),'cache':ps(slots),'pending':state(inputs,'Input tokens are consumed before their KV is appended'),'growth':dict(series=[dict(id='linear',label='Declared toy shape',role='reference',points=[[0,0],[8,8*bpt]])],markers=[dict(id='current',x=count,y=count*bpt,label='Current allocation')],status='No runtime workspace is modeled')}
 ex={'empty':'The toy model has two layers, one KV head, two elements per head, and two bytes per element.','weights':'Load the fixed model weights; later request growth does not change their occupancy.','prefillA':'Prefill stores two prompt positions owned by A.','prefillB':'Prefill stores two prompt positions owned by B.','consume':'Each active request consumes its previously emitted token; its KV is not allocated yet.','append':'Compute and append KV for the consumed input of each active request in parallel.','complete':'A finishes its last access. Its storage remains allocated until release.','release':'Release only A-owned KV after its last consumer. B remains resident.'}[id]
 steps.append(step(id,{'empty':'Declare the toy capacity','weights':'Load fixed weights','prefillA':'Prefill request A','prefillB':'Prefill request B','consume':'Consume the next decode inputs','append':'Append current-input KV','complete':'Complete request A','release':'Release A-owned cache'}[id],ex,o,[metric('occupied','Retained KV positions',count,'positions'),metric('bytes','KV footprint',count*bpt,'B')],{'empty':'Each consumed token will add the same per-layer key/value footprint.','weights':'Request growth draws on capacity left after the fixed weights.','prefillA':'A retains its prompt state for later decode reads.','prefillB':'B adds its own state without replacing A’s retained context.','consume':'Current input tokens are visible, while their KV positions remain unallocated.','append':'Both current inputs now have retained KV; subsequent queries can reuse it.','complete':'A has no remaining consumer, so its owned storage is eligible for release.','release':'A’s positions become free while B’s context and the model weights remain resident.'}[id],None if id=='empty' else 'release' if id=='release' else 'execute' if id=='complete' else 'write',['cache'] if id not in ['weights','consume','complete'] else ['weights'] if id=='weights' else ['pending']))
lesson('fig-kv-cache-growth','Request growth consumes the remaining capacity','How does each request change KV residency while weights remain fixed?', 'Explicit toy shape, not the plotted Llama model: two layers, one KV head, head dimension two, two-byte KV elements. Two model-weight slots are abstract; no physical device or runtime workspace is modeled.','ownership',objects,steps,['KV bytes per token remain constant for the declared shape.','Consumed decode inputs receive KV only at append.','Completing a request does not free another request’s storage.'],[{'kind':'metricEqualsOccupied','metric':'occupied','object':'cache'}])
# Chunked admission, one bounded iteration per action.
objects=[obj('worker','lanes','One worker: prompt and active decode','compute',{'lanes':['Prompt P','Active stream'],'start':0,'end':12,'axisLabel':'Schematic iteration work'},0),seq('prompt','Processed prompt chunks','input',4,1),seq('tokens','Emitted active-stream tokens','output',5,2)]
def prefillcase(chunked):
 tasks=[]
 if chunked:
  for i in range(4):tasks += [dict(id='p'+str(i),lane=0,start=i*3,end=i*3+2,label='P'+str(i),role='compute'),dict(id='d'+str(i),lane=1,start=i*3+2,end=i*3+3,label='Decode',role='compute')]
 else:tasks=[dict(id='whole',lane=0,start=0,end=8,label='Whole P',role='compute'),dict(id='decode',lane=1,start=8,end=9,label='Decode',role='compute')]
 steps=[]
 # Full prompt then one decode gives only 3 snapshots: initial prior token establishes state, admit whole, decode, next decode.
 if not chunked:tasks.append(dict(id='decode2',lane=1,start=9,end=10,label='Decode',role='compute'))
 for done in range(len(tasks)+1):
  prompt=[None]*4;tokens=['t0']+[None]*4
  completed=tasks[:done];n=0
  for task in completed:
   if task['lane']==0:
    if chunked:prompt[int(task['id'][1:])]=task['id']
    else:prompt=['p'+str(j) for j in range(4)]
   else:n+=1;tokens[n]='t'+str(n)
  segs=[dict(t,state='done' if i<done else 'waiting') for i,t in enumerate(tasks)]
  o={'worker':dict(segments=segs,cursor=completed[-1]['end'] if done else 0,status='Each boundary permits a scheduling decision'),'prompt':state(prompt),'tokens':state(tokens,'t0 predates admission')}
  ex='Prompt chunks and decode steps share one worker; no overlap is implied.' if not done else 'This action completes one prefill chunk.' if completed[-1]['lane']==0 and chunked else 'This action completes the admitted whole prompt.' if completed[-1]['lane']==0 else 'The iteration boundary allows one active-stream decode and one new output.'
  steps.append(step('ready' if not done else 'task'+str(done),'Establish the active stream' if not done else 'Complete '+completed[-1]['label'],ex,o,[metric('prompt','Processed prompt chunks',sum(v is not None for v in prompt),'chunks'),metric('emitted','New decode outputs',n,'tokens')], ('The active stream keeps t0 while prompt P waits for admission.' if not done else 'The completed prompt work leaves the next iteration boundary available to decode.' if completed[-1]['lane']==0 else 'A new token appears without discarding the stream’s earlier outputs.'),None if not done else 'execute',[] if not done else [completed[-1]['id'],'prompt','tokens']))
 return steps
lesson('fig-serving-chunked-prefill','Admission boundaries govern decode opportunities','When can the existing stream advance during a new prompt?', 'Four equal schematic prompt chunks and one active stream on one worker. Work units are not measured durations; chunking creates scheduling opportunities, not a throughput guarantee.','schedule',objects,prefillcase(True),['One worker executes a prefill or decode task at a time.','Both policies process the same four prompt chunks.','Existing output tokens remain visible.'],variants=[dict(id='whole',label='Whole-prompt admission',scope='Same four prompt chunks admitted as one prefill task. Two later decode steps make subsequent cadence visible.',steps=prefillcase(False))])
# Paged mapping, exact source identities and last-block occupancy.
objects=[seq('logical','Request E: logical pages in token order','input',4,0),seq('table','E block table: logical index → physical block','control',4,1),pool('physical','Eight physical KV blocks',8,2),obj('reader','chamber','Query, resolved block, and KV read','compute',{'slots':3},3)]
initial=[None,None,slot('C','C'),slot('C','C'),None,None,slot('D','D'),slot('D','D')];physical=copy.deepcopy(initial);logical=[None]*4;table=[None]*4;reader=['Query E:p2',None,None];steps=[]
requested_pages=4; pending_readers=0; reader_finished=False
for id in ['free','allocate','lookup','read','finish','release']:
 if id=='allocate':
  logical=['p'+str(j)+(':partial' if j==requested_pages-1 else ':full') for j in range(requested_pages)];table=[i for i,s in enumerate(physical) if s is None][:requested_pages]
  assert len(table)==requested_pages
  for j,b in enumerate(table):physical[b]=slot('E:p'+str(j)+(' partial' if j==3 else ''),'E')
 if id=='lookup':reader[1]=table[2]
 if id=='read':
  reader[2]='E:p2 KV';pending_readers+=1
 if id=='finish':
  pending_readers-=1;reader_finished=True;reader=['E complete',None,None]
 if id=='release':
  assert reader_finished and pending_readers==0, 'E still has an outstanding consumer'
  physical=[None if s is not None and s['owner']=='E' else s for s in physical];logical=[None]*requested_pages;table=[None]*requested_pages
 assert all(physical[i]==s for i,s in enumerate(initial) if s is not None), 'Other request ownership changed'
 count=sum(s is not None for s in physical)
 o={'logical':state(logical),'table':state(table),'physical':ps(physical),'reader':state(reader)}
 tokens=[]
 if id=='read':tokens=[dict(id='readE2',label='E:p2',value='E:p2 KV',role='input',owner='E',at={'object':'reader','index':2},state='copy',sourceAt={'object':'physical','index':reader[1]})]
 ex={'free':'C and D leave free blocks in two separated runs. E needs four equal-size blocks.','allocate':'Assign E logical pages to physical blocks 0, 1, 4, and 5. The partially filled last block remains allocated.','lookup':'Resolve logical page 2 through E’s table entry 2; it names physical block 4.','read':'Read physical block 4 using the resolved table value. The source block remains resident.','finish':'The final E access finishes. No in-flight E reader remains, but E still owns its blocks.','release':'Return only E’s unshared blocks to the free pool and invalidate E’s mappings.'}[id]
 steps.append(step(id,{'free':'Inspect the fragmented free pool','allocate':'Allocate E’s four page mappings','lookup':'Resolve E logical page 2 to block 4','read':'Read E logical page 2','finish':'Finish E’s pending access','release':'Release E’s blocks and mappings'}[id],ex,o,[metric('occupied','Allocated physical blocks',count,'blocks'),metric('free','Free physical blocks',len(physical)-count,'blocks')], {'free':'Four free blocks exist, but they do not form one contiguous run.','allocate':'E’s logical order is preserved even though its physical blocks are separated.','lookup':'Logical page 2 skips C’s blocks because E’s table names block 4.','read':'The query receives a copy; E’s original block remains allocated.','finish':'E can now release its blocks because its pending access has finished.','release':'C and D retain their original blocks; E’s former blocks can serve another request.'}[id],None if id=='free' else 'copy' if id=='read' else 'release' if id=='release' else 'execute' if id in ['lookup','finish'] else 'write',['reader'] if id in ['lookup','finish'] else ['physical'] if id!='read' else ['readE2'],tokens=tokens))
lesson('fig-serving-paged-kv-mapping','Logical order survives scattered placement','How does an ordered sequence use noncontiguous blocks safely?', 'Source eight-block pool, equal block sizes, requests C/D/E, no prefix sharing. E has four pages and a partially filled final block; paging does not change KV bytes per token.','ownership',objects,steps,['The E table maps logical page 2 to physical block 4.','A read preserves the block and its ownership.','E blocks become free only after its last access; C and D never move.'],[{'kind':'metricEqualsOccupied','metric':'occupied','object':'physical'}])
# Two-request pipeline, fixed stage durations and explicit ready/output surfaces.
objects=[obj('schedule','lanes','CPU preparation and GPU execution','compute',{'lanes':['CPU','GPU'],'start':0,'end':6,'axisLabel':'Toy work units'},0),pool('ready','Prepared inputs awaiting GPU',2,1),seq('output','Completed request outputs','output',2,2)]
def pipelinecase(pipelined):
 tasks=[dict(id='cpuA',lane=0,start=0,end=2,label='A prepare',role='input'),dict(id='gpuA',lane=1,start=2,end=4,label='A GPU',role='compute'),dict(id='cpuB',lane=0,start=2 if pipelined else 4,end=4 if pipelined else 6,label='B prepare',role='input'),dict(id='gpuB',lane=1,start=4 if pipelined else 6,end=6 if pipelined else 8,label='B GPU',role='compute')]
 # Shared axes must include the serial endpoint.
 steps=[];ready=[None,None];out=[None,None]
 for stage in ([0,1,3,4] if pipelined else range(5)):
  if stage==1:ready[0]=slot('A input','A','ready')
  if stage==2:ready[0]=None;out[0]='A result'
  if stage==3:
   ready[1]=slot('B input','B','ready')
   if pipelined:ready[0]=None;out[0]='A result'
  if stage==4:ready[1]=None;out[1]='B result'
  # CPU B completion is separate from GPU A even when both occur at t=4.
  completed=set(['cpuA','gpuA','cpuB','gpuB'][:stage]);segs=[dict(t,state='done' if t['id'] in completed else 'waiting') for t in tasks]
  o={'schedule':dict(segments=segs,status='Independent resources may overlap; task completion steps are inspectable.'),'ready':ps(ready),'output':state(out)}
  steps.append(step('ready' if not stage else 'finish'+str(stage),('Finish A GPU and B preparation in parallel' if pipelined and stage==3 else ['Establish two independent requests','Finish A preparation','Finish A GPU execution','Finish B preparation','Finish B GPU execution'][stage]),['Both requests use the same preparation and inference work.','A input is ready before its GPU stage can execute.','A GPU execution consumes its prepared input and produces A’s output.','B preparation uses the CPU, which can work while A occupies the GPU.','B GPU execution waits for both its input and the GPU resource.'][stage],o,[metric('completed','Completed requests',sum(v is not None for v in out),'requests'),metric('makespan','Declared schedule endpoint',max(t['end'] for t in tasks),'work units')], ('Both requests require their own preparation before inference.' if not stage else 'A’s GPU stage is now eligible because its input is ready.' if stage==1 else 'A’s output is available.' if stage==2 else 'B’s input is ready while A’s output is available; B may use the freed GPU.' if stage==3 else 'Both outputs are complete; overlapping independent stages shortened the schedule.' if pipelined else 'Both outputs are complete after serially executing the same stage work.'),None if not stage else 'execute',[] if not stage else (['gpuA','cpuB'] if pipelined and stage==3 else [['cpuA'],['gpuA'],['cpuB'],['gpuB']][stage-1])))
 return steps
objects[0]['config']['end']=8
lesson('fig-serving-pipeline-timing','Prepare B while A uses the accelerator','Which dependency permits overlap without changing request identity?', 'Two toy requests, CPU preparation and GPU inference each take two schematic work units. Transfer and postprocessing are outside this focused two-stage view. A GPU and B CPU complete together at the shared boundary.','schedule',objects,pipelinecase(True),['GPU execution requires the same request’s prepared input.','One CPU and one GPU each execute at most one task at a time.','Stage durations remain identical across policies.'],variants=[dict(id='serial',label='Serial requests',scope='Same toy stages; B preparation starts only after A GPU execution completes.',steps=pipelinecase(False))])
# Node residency, exact chapter-derived slot counts; no hardware sizes are invented.
objects=[pool('node','Eight accelerator slots on one node',8,0),seq('artifact','Persistent candidate artifact outside accelerator slots','output',1,1),seq('readers','Outstanding retraining / serving use','control',2,2)]
base=[slot('Serving','Serving')]*2+[slot('Hot standby','Rollback')]*2+[None]*4;slots=copy.deepcopy(base);artifact=[None];readers=['Serving active',None];steps=[]
for id in ['ready','train','finish','release','canary']:
 if id=='train':slots[4:]=[slot('Training state','Retrain') for _ in range(4)];readers[1]='Retrain active'
 if id=='finish':artifact=['Candidate persisted'];readers[1]='Retrain complete'
 if id=='release':slots[4:]=[None]*4
 if id=='canary':slots[4:6]=[slot('Candidate canary','Canary') for _ in range(2)]
 n=sum(s is not None for s in slots);o={'node':ps(slots),'artifact':state(artifact),'readers':state(readers)}
 ex={'ready':'Serving and the immediate-rollback copy are already resident.','train':'The retraining run occupies the remaining four slots and leaves no room for another loaded copy.','finish':'Retraining persists its candidate outside accelerator residency. Its device allocations still require release.','release':'The completed retraining job releases its four slots after its last device access.','canary':'Load the candidate canary into two freed slots; the existing hot standby remains available.'}[id]
 steps.append(step(id,{'ready':'Keep serving and rollback resident','train':'Allocate the retraining job','finish':'Persist the completed candidate','release':'Release completed training allocations','canary':'Load the candidate canary'}[id],ex,o,[metric('occupied','Occupied accelerator slots',n,'slots'),metric('free','Free accelerator slots',len(slots)-n,'slots')], {'ready':'Immediate rollback reserves resident capacity before retraining begins.','train':'The node is full; another loaded copy cannot fit while retraining runs.','finish':'The candidate exists persistently, but training still occupies its device slots.','release':'Freed training slots can now hold a candidate without evicting the hot standby.','canary':'Canary and hot standby coexist; two slots remain free under this schedule.'}[id],None if id=='ready' else 'release' if id=='release' else 'execute' if id=='finish' else 'write',['node'] if id not in ['finish'] else ['artifact','readers']))
lesson('fig-ml-ops-dlrm-node-day','Rollback consumes resident capacity','Why can retraining block an additional loaded model copy?', 'Source node has eight accelerator slots: serving and immediate rollback use two each; retraining uses four, then a canary uses two. The chapter’s six-hour run duration is outside this capacity-only lifecycle view.','ownership',objects,steps,['Serving and hot standby remain resident through the whole schedule.','An on-disk candidate does not itself occupy accelerator slots.','Completed training allocations must be released before a canary loads.'],[{'kind':'metricEqualsOccupied','metric':'occupied','object':'node'}])
if __name__=='__main__':
 import sys
 target=pathlib.Path(__file__).with_name('vol1-c.json')
 if '--write' in sys.argv:
  target.write_text(json.dumps(L,ensure_ascii=False,indent=2)+'\n')
 else:
  actual=json.loads(target.read_text())
  assert actual==L, 'Authored JSON differs from recomputed declared-input states.'
  for item in actual:
   for case,steps in [('base',item['steps'])]+[(v['id'],v['steps']) for v in item.get('variants',[])]:
    final=steps[-1]
    print(item['key'],case,final['id'],json.dumps(final['snapshot']['metrics'],ensure_ascii=False))
  print('PASS: every snapshot matches the recomputed declared-input model; no files rewritten.')

#!/usr/bin/env python3
import json, copy, pathlib
OUT=pathlib.Path(__file__).resolve().parent
def obj(i,k,label,role,config,n):
 return dict(id=i,kind=k,label=label,role=role,config=config,layout={'wide':{'row':n//2,'column':n%2},'narrow':{'row':n,'column':0}})
def pool(i,label,slots,n,labels=None,role='memory'):
 return obj(i,'pool',label,role,dict(slots=slots,columns=min(slots,4),slotLabels=labels or [str(x) for x in range(slots)]),n)
def reg(i,label,n,typ='integer',unit=''):
 return obj(i,'register',label,'control',dict(type=typ,unit=unit),n)
def ps(values,status='',owner='Lesson',state='retained'):
 return dict(slots=[None if v is None else dict(value=v,owner=owner,status=state) for v in values],status=status)
def rs(v,status='',previous=None,expression=None):
 d=dict(value=v,status=status)
 if previous is not None:d['previous']=previous
 if expression is not None:d['expression']=expression
 return d
def metric(i,label,v,unit=''):
 return dict(id=i,label=label,value=v,unit=unit)
def tr(kind,subjects,meaning,fr=None,to=None):
 d=dict(kind=kind,subjects=subjects,meaning=meaning)
 if fr is not None:d['from']=fr
 if to is not None:d['to']=to
 return d
def step(i,title,ex,states,metrics,result,transition=None,tokens=None,active=None):
 return dict(id=i,title=title,explanation=ex,snapshot=dict(objects=copy.deepcopy(states),tokens=tokens or [],metrics=metrics,activeConnections=active or [],result=result),transition=transition)
def lesson(key,title,q,scope,family,objects,steps,invariants,**kw):
 return dict(schema=1,key='vol2:'+key,title=title,question=q,scope=scope,family=family,objects=objects,steps=steps,invariants=invariants,**kw)
def cp(i,value,fr,to,owner='Tensor T'):
 return dict(id=i,label='T',value=value,role='communication',owner=owner,at=to,state='copy',sourceAt=fr)
def lanes(i,label,n,names,end):
 return obj(i,'lanes',label,'compute',dict(lanes=names,start=0,end=end,axisLabel='Schematic progress'),n)
def seg(i,l,a,b,label,state='waiting',role='compute'):
 return dict(id=i,lane=l,start=a,end=b,label=label,state=state,role=role)
def ls(segments,status='',cursor=None):
 return dict(segments=copy.deepcopy(segments),status=status,cursor=cursor)

def gpu():
 M=8; payload=f'T ({M} bytes)'
 objects=[pool('gpu','Source GPU memory',1,0,['T']),pool('host','Registered host buffer',1,1,['T']),pool('nic','RDMA NIC',1,2,['T'],'communication'),pool('fabric','Fabric payload',1,3,['T'],'communication')]
 def case(direct):
  s={'gpu':ps([payload],owner='Tensor T'),'host':ps([None]),'nic':ps([None]),'fabric':ps([None])};steps=[]
  def add(i,title,ex,host,wire,transition=None,tokens=None):
   steps.append(step(i,title,ex,s,[metric('hostTraffic','Host-memory payload traffic',host,'bytes'),metric('wirePayload','Payload sent to fabric',wire,'bytes')],f'Host-memory traffic: {host} bytes; fabric payload: {wire} bytes.',transition,tokens))
  add('source','Retain the source tensor','The same tensor T is used in both declared paths.',0,0)
  if direct:
   s['nic']=ps([payload],owner='Tensor T',state='ready');add('peer-read','Read GPU memory into the NIC','The PCIe peer path bypasses the registered host buffer.',0,0,tr('copy',['gpu'],'Read GPU memory without consuming its tensor.',{'object':'gpu','index':0},{'object':'nic','index':0}),[cp('peer-copy',payload,{'object':'gpu','index':0},{'object':'nic','index':0})])
   s['fabric']=ps([payload],owner='Tensor T',state='ready');add('send','Send T onto the fabric','The network still carries the full payload.',0,M,tr('copy',['nic'],'Transmit one payload; retain the sender copy.',{'object':'nic','index':0},{'object':'fabric','index':0}),[cp('wire-copy',payload,{'object':'nic','index':0},{'object':'fabric','index':0})])
   s['nic']=ps([None]);add('release','Release the completed NIC staging slot','The fabric has accepted the payload; the source GPU tensor remains retained.',0,M,tr('release',['nic'],'Release the completed send buffer.'))
  else:
   s['host']=ps([payload],owner='Tensor T');add('stage','Write T into registered host memory','This payload write contributes one payload length of host-memory traffic.',M,0,tr('copy',['gpu'],'Stage a copy without consuming the GPU source.',{'object':'gpu','index':0},{'object':'host','index':0}),[cp('host-copy',payload,{'object':'gpu','index':0},{'object':'host','index':0})])
   s['nic']=ps([payload],owner='Tensor T',state='ready');add('read','Read the host copy into the NIC','The NIC read adds a second payload length of host-memory traffic.',sum([M,M]),0,tr('copy',['host'],'Read the retained host copy.',{'object':'host','index':0},{'object':'nic','index':0}),[cp('nic-copy',payload,{'object':'host','index':0},{'object':'nic','index':0})])
   s['fabric']=ps([payload],owner='Tensor T',state='ready');add('send','Send T onto the fabric','The fabric carries the same tensor as the direct path.',sum([M,M]),M,tr('copy',['nic'],'Transmit the full payload.',{'object':'nic','index':0},{'object':'fabric','index':0}),[cp('wire-copy',payload,{'object':'nic','index':0},{'object':'fabric','index':0})])
  return steps
 return lesson('fig-gpudirect-data-path','Follow one tensor through the sending node','Which memory traffic disappears when the NIC reads GPU memory directly?','Toy sending-node transfer of an 8-byte payload. Count only payload writes and reads at host memory; exclude protocol headers, transactions, receiving-node traffic, and hardware timing.', 'communication',objects,case(False),['Reads preserve the source tensor and any retained host copy.','Both paths send the same payload onto the network.','Removing host staging does not change the physical network rate.'],variants=[dict(id='direct',label='GPUDirect peer path',scope='Same tensor, direct NIC read from GPU memory; identical host and fabric accounting.',steps=case(True))])

def pfc():
 objects=[pool('b'+str(i),f'Switch {i} queue b{i}',1,i-1,[f'next b{(i%4)+1}']) for i in range(1,5)]
 objects +=[obj('pause','sequence','PAUSE messages sent upstream','control',dict(slots=4,slotLabels=['from b1','from b2','from b3','from b4']),4),reg('drops','Watchdog packet drops',5)]
 s={f'b{i}':ps([f'P{i} → b{(i%4)+1}' if i<4 else None],owner=f'Queue b{i}') for i in range(1,5)};s['pause']={'values':[None]*4,'status':'No PAUSE emitted yet'};s['drops']=rs(0)
 steps=[]
 def add(i,title,ex,t=None):
  count=sum(v['slots'][0] is not None for k,v in s.items() if k.startswith('b'))
  steps.append(step(i,title,ex,s,[metric('resident','Resident packets',count,'packets'),metric('dropped','Packets deliberately dropped',s['drops']['value'],'packets')],f'{count} packets remain resident; {s["drops"]["value"]} packet has been dropped.' if s['drops']['value'] else f'{count} packets remain resident; no packet has been dropped.',t))
 add('open','Begin with one free downstream slot','Each packet requires the next queue clockwise. The toy queue capacity is one packet.')
 s['b4']=ps(['P4 → b1'],owner='Queue b4');add('close','Fill the last queue','P4 closes the occupied dependency cycle; no packet has downstream space.',tr('write',['b4'],'Enqueue P4 into the final free slot.',to={'object':'b4','index':0}))
 s['pause']={'values':['PAUSE b4','PAUSE b1','PAUSE b2','PAUSE b3'],'status':'Counterclockwise PAUSE; clockwise packet dependencies'};add('pause','Assert PAUSE toward every upstream predecessor','Each full queue pauses its predecessor. All four packets remain resident and none can drain.',tr('reveal',['pause'],'Emit one parallel round of upstream PAUSE control messages.'))
 s['b1']=ps([None]);s['drops']=rs(1,expression='0 + 1');add('drop','Drop P1 under the declared watchdog policy','The external watchdog deliberately sacrifices one packet to open space; transport recovery is outside this scene.',tr('release',['b1'],'Watchdog drops P1 rather than delivering it.'))
 s['pause']['values'][0]=None;s['pause']['status']='b1 has space and clears its PAUSE toward b4';add('resume','Clear b1’s PAUSE toward b4','Space alone does not override a still-active PAUSE. Queue b1 now allows its upstream neighbor to send.',tr('write',['pause'],'Clear only the PAUSE emitted by b1.',to={'object':'pause','index':0}))
 s['b4']=ps([None]);s['b1']=ps(['P4 → onward'],owner='Queue b1');s['pause']['status']='Earlier PAUSE clearance allowed b4 to forward; later controls are outside this scene';add('forward','Forward P4 from b4 into the freed slot','One packet advances clockwise. Further PAUSE clearance and reliable recovery follow outside this bounded scene.',tr('move',['b4','b1'],'Forward P4 after downstream space and PAUSE clearance.',{'object':'b4','index':0},{'object':'b1','index':0}))
 return lesson('fig-pfc-credit-deadlock','Break a cyclic buffer dependency','Why do full queues and upstream PAUSE prevent every packet from advancing?','Four queues with one packet slot each, abstracting the full queues in the figure. Initial packets require the next queue clockwise. One explicit watchdog drop and one subsequent hop are shown; no timer, retransmission, or hardware policy is modeled.','control',objects,steps,['PAUSE travels upstream, opposite to the packet-forwarding dependency.','No packet leaves a full cycle before the external watchdog drop.','Resident packets plus deliberate drops equal all packets admitted; a drop is not successful delivery.'])

def hierarchy():
 inputs=[[1,2],[3,4],[5,6],[7,8]]
 objects=[pool('r'+str(r),f'Node {r//2} · GPU {r%2}',2,r,['Shard A','Shard B']) for r in range(4)]
 steps=[]
 def snap(i,title,ex,values,phase,t=None,tokens=None):
  s={f'r{r}':ps(values[r],owner=f'Node {r//2} GPU {r%2}',state='ready' if phase>=2 else 'retained') for r in range(4)}
  held=sum(x is not None for row in values for x in row)
  steps.append(step(i,title,ex,s,[metric('held','Resident shard slots',held,'shards'),metric('phase','Completed collective phases',phase,'phases')],('Every GPU now holds both global sums.' if phase==3 else 'Shard ownership follows the tier used by this phase.'),t,tokens))
 snap('inputs','Start with four rank contributions','Each GPU contributes two scalar shards. Corresponding shard indices must be added together.',inputs,0)
 local=[[None]*2 for _ in inputs]
 for node in range(2):
  for c in range(2):local[node*2+c][c]=sum(inputs[r][c] for r in range(node*2,node*2+2))
 snap('local','ReduceScatter within each node','A local collective adds corresponding contributions and leaves each GPU responsible for one node-partial shard.',local,1,tr('execute',['r0','r1','r2','r3'],'Complete the two independent node-local ReduceScatter operations.'))
 global_sums=[sum(row[c] for row in inputs) for c in range(2)];global_shards=[[None]*2 for _ in inputs]
 for r in range(4):global_shards[r][r%2]=global_sums[r%2]
 snap('global','AllReduce corresponding shards across nodes','GPU 0 peers combine shard A; GPU 1 peers combine shard B. Each global sum is replicated only across its shard peers.',global_shards,2,tr('execute',['r0','r1','r2','r3'],'Complete one inter-node AllReduce for each disjoint shard.'))
 snap('gather','AllGather the finished shards inside each node','The local AllGather distributes finished sums without adding them again.',[global_sums[:] for _ in inputs],3,tr('copy',['r0','r1','r2','r3'],'Copy the complementary finished shard to the other GPU in each node.'),[dict(id=f'gather{r}',label='A' if (1-r%2)==0 else 'B',value=global_sums[1-r%2],role='communication',owner=f'Node {r//2} GPU {r%2}',at={'object':f'r{r}','index':1-r%2},state='copy',sourceAt={'object':f'r{r^1}','index':1-r%2})for r in range(4)])
 return lesson('fig-hierarchical-allreduce','Track shard ownership across the hierarchy','How does each GPU reach the full sum while inter-node work stays sharded?','Toy two-node system with two GPUs per node and two scalar shards per rank. Each step completes one logical collective phase; micro-rounds, bytes, bandwidth, and latency are outside the model.','reduction',objects,steps,['Local reduction adds only contributions from the same node.','Inter-node peers reduce corresponding shard indices, never different shards.','AllGather copies finished sums and performs no additional summation.'])

def ring():
 N=4;C=4;inputs=[[10*(r+1)+c+1 for c in range(C)]for r in range(N)]
 values=copy.deepcopy(inputs);prov=[[{r} for c in range(C)]for r in range(N)];objects=[pool('r'+str(r),f'GPU {r} · sends to GPU {(r+1)%N}',C,r,list('ABCD'))for r in range(N)];steps=[]
 for ob in objects:ob['config']['columns']=2
 def snap(i,title,ex,rounds,t=None,tokens=None):
  s={}
  for r in range(N):
   slots=[]
   for c in range(C):
    val=values[r][c];who=','.join(map(str,sorted(prov[r][c])))
    slots.append(dict(value=f'{val} [{who}]',owner=f'GPU {r}',status='ready' if len(prov[r][c])==N else 'retained'))
   s[f'r{r}']=dict(slots=slots,status='Value [contributing ranks]; green/ready only when all ranks contributed')
  complete=sum(len(p)==N for row in prov for p in row)
  steps.append(step(i,title,ex,s,[metric('rounds','Completed ring rounds',rounds,'rounds'),metric('sent','Chunks sent per rank',rounds,'chunks'),metric('complete','Fully reduced resident chunks',complete,'chunks')],f'{complete} resident chunks contain all rank contributions; each rank has sent {rounds} chunks.',t,tokens))
 snap('inputs','Start with indexed local contributions','Every GPU holds A through D. Brackets name the ranks already included in each numeric value.',0)
 for rd in range(N-1):
  old=copy.deepcopy(values);oldp=copy.deepcopy(prov)
  for r in range(N):
   sender=(r-1)%N;c=(r-rd-1)%N
   assert not (prov[r][c]&oldp[sender][c])
   values[r][c]+=old[sender][c];prov[r][c]|=oldp[sender][c]
  snap('reduce'+str(rd+1),f'ReduceScatter round {rd+1}','Each rank sends one partial chunk clockwise; the receiver adds only contributions not already present.',rd+1,tr('execute',[f'r{r}'for r in range(N)],'Complete one simultaneous send/receive-and-add ring round.'))
 assert all(len(prov[r][(r+1)%N])==N for r in range(N))
 for rd in range(N-1):
  old=copy.deepcopy(values);oldp=copy.deepcopy(prov)
  for r in range(N):
   sender=(r-1)%N;c=(r-rd)%N
   assert len(oldp[sender][c])==N
   values[r][c]=old[sender][c];prov[r][c]=oldp[sender][c].copy()
  snap('gather'+str(rd+1),f'AllGather round {rd+1}','Each rank forwards one finished chunk clockwise. Receivers overwrite stale partials with the finished sum; they do not add again.',N-1+rd+1,tr('copy',[f'r{r}'for r in range(N)],'Copy one finished chunk from each rank to its clockwise neighbor.'),[dict(id=f'copy{rd}-{r}',label='ABCD'[(r-rd)%N],value=values[r][(r-rd)%N],role='communication',owner=f'GPU {r}',at={'object':f'r{r}','index':(r-rd)%N},state='copy',sourceAt={'object':f'r{(r-1)%N}','index':(r-rd)%N})for r in range(N)])
 assert all(values[r]==[sum(inputs[k][c] for k in range(N))for c in range(C)]for r in range(N))
 return lesson('fig-ring-allreduce','Follow values around the ring','When does a partial chunk become a finished sum, and why does AllGather stop adding?','Four ranks, four scalar chunks A–D per rank, clockwise logical ring with GPU 3 wrapping to GPU 0. Brackets show exact contributing rank IDs. One step is one simultaneous logical ring round; no physical topology or timing is inferred.','reduction',objects,steps,['ReduceScatter combines disjoint contributor sets for the same chunk index.','After three ReduceScatter rounds GPU 0 owns B, GPU 1 owns C, GPU 2 owns D, and GPU 3 owns A as finished sums.','AllGather copies fully reduced chunks; it never adds a finished sum to an old partial.','Each rank sends one chunk per round and retains the full replicated sum at completion.'])

def checkpoint():
 base=10;live=base;advance=1;new_live=base+advance
 objects=[pool('local','Immutable local NVMe shards',2,0,['Rank 0','Rank 1']),pool('durable','Parallel file system generations',3,1,['g0 committed','g1 rank 0','g1 rank 1']),reg('live','Live training weight',2,'toy scalar','weight units'),obj('verified','sequence','g1 shard checksum results','control',dict(slots=2,slotLabels=['Rank 0','Rank 1']),3),reg('manifest','Committed restore generation',4,'generation'),lanes('timeline','Checkpoint and training dependencies',5,['Training','Local NVMe','Parallel FS','Coordinator'],8)]
 tasks=[seg('pause',0,0,1,'Pause / consistent cut','running','control'),seg('train',0,1,8,'Live training'),seg('write',1,0,1,'Write g1','running','memory'),seg('copy0',2,1,3,'Copy rank 0',role='communication'),seg('copy1',2,3,5,'Copy rank 1',role='communication'),seg('verify',3,5,6,'Verify shards',role='control'),seg('publish',3,6,7,'Publish manifest',role='control')]
 s={'local':ps([None,None]),'durable':ps(['g0 complete',None,None],owner='Generation g0'),'live':rs(live),'verified':{'values':[None,None],'status':'Not verified'},'manifest':rs('g0'),'timeline':ls(tasks)};steps=[]
 def add(i,title,ex,t=None,tokens=None):
  ready=sum(v is not None for v in s['durable']['slots'][1:]);verified=sum(v is not None for v in s['verified']['values'])
  steps.append(step(i,title,ex,s,[metric('durableShards','g1 durable shards',ready,'shards'),metric('verifiedShards','g1 verified shards',verified,'shards'),metric('restore','Restore generation',s['manifest']['value'])],f'Restore uses {s["manifest"]["value"]}; live weight is {s["live"]["value"]}.',t,tokens))
 def ts(ids,state):
  for t in tasks:
   if t['id'] in ids:t['state']=state
  s['timeline']=ls(tasks)
 add('cut','Pause at a consistent training cut','The previous committed generation g0 remains the only restore point.')
 shard=[f'g1:r{r} w={base}'for r in range(2)]
 s['local']=ps(shard,owner='Generation g1');ts(['pause','write'],'done');ts(['train','copy0'],'running');add('stage','Finish immutable local shard writes','Both shards describe the same consistent cut. Training may now resume, but g1 is not durably committed.',tr('write',['local'],'Finish parallel local writes from one consistent training cut.',to={'object':'local'}))
 s['live']=rs(new_live,previous=base,expression=f'{base} + {advance}');add('train','Advance live training on its next step','Live state changes while local g1 shards retain the earlier values used by the checkpoint.',tr('execute',['live','train'],'Complete one live training step while the immutable copy source is retained.'))
 for r in range(2):
  s['durable']['slots'][r+1]=dict(value=shard[r],owner='Generation g1',status='retained');ts(['copy'+str(r)],'done')
  if r==0:ts(['copy1'],'running')
  else:ts(['verify'],'running')
  add('copy'+str(r),f'Finish the durable copy of rank {r}’s shard','A completed shard copy is durable data, not a published checkpoint.',tr('copy',['local'],'Persist one immutable shard without consuming local staging.',{'object':'local','index':r},{'object':'durable','index':r+1}),[dict(id='shard-copy'+str(r),label=f'g1:r{r}',value=shard[r],role='memory',owner='Generation g1',at={'object':'durable','index':r+1},state='copy',sourceAt={'object':'local','index':r})])
 s['verified']={'values':['match','match'],'status':'Every persisted shard matches its declared checksum'};ts(['verify'],'done');ts(['publish'],'running');add('verify','Verify every required shard','The coordinator checks the complete shard set before publication. The restore pointer still names g0.',tr('execute',['verified','verify'],'Verify the two shard checksums as one parallel verification operation.'))
 s['manifest']=rs('g1',previous='g0');ts(['publish'],'done');add('publish','Publish the g1 manifest atomically','Only the manifest commit makes g1 the declared restore point; staged data alone never changes that pointer.',tr('write',['manifest'],'Atomically publish the complete verified generation.',to={'object':'manifest'}))
 s['local']=ps([None,None]);add('release','Release local staging after publication','The durable complete generation is committed and no modeled consumer still needs the local shards.',tr('release',['local'],'Free local staging after its copy and commit obligations.'))
 return lesson('fig-storage-checkpoint-staging','Separate staging from checkpoint commitment','When can a resumed training run safely restore from the new generation?','Toy two-rank generation g1 captures scalar weight 10 while g0 is already committed. One later training step adds one. Schematic lanes show dependencies rather than measured duration; checksum match is an explicit verification outcome, not a simulated hash algorithm.','ownership',objects,steps,['Every g1 shard belongs to the same immutable consistent cut.','Live training cannot mutate an already staged g1 shard.','The restore pointer remains g0 until all g1 shards are durable, verified, and named by the atomic manifest.','No local staging slot is released before copy completion and manifest publication.'])

def overlap():
 b=3;o=1;net=3;boundary=b+o;last_done=boundary+b;last_launch=last_done+o;tail_end=last_launch+net;end=tail_end+1
 objects=[lanes('timeline','Final backward layers and their AllReduce',0,['GPU','Network'],end),pool('grads','Gradient readiness and reduced results',2,1,['Layer 2','Layer 1']),reg('weight','Optimizer state',2,'toy weight','weight units')]
 tasks=[seg('b2',0,0,b,'Backward L2','running'),seg('o2',0,b,boundary,'Launch overhead L2',role='control'),seg('b1',0,boundary,last_done,'Backward L1'),seg('o1',0,last_done,last_launch,'Launch overhead L1',role='control'),seg('idle',0,last_launch,tail_end,'Input to optimizer: wait',role='control'),seg('opt',0,tail_end,end,'Optimizer update'),seg('a2',1,boundary,boundary+net,'AllReduce L2',role='communication'),seg('a1',1,last_launch,tail_end,'AllReduce L1',role='communication')]
 s={'timeline':ls(tasks,cursor=0),'grads':ps([None,None]),'weight':rs(10)};steps=[];ready=0;reduced=0;exposed=0
 def ts(done=(),running=(),cursor=None):
  for t in tasks:
   if t['id'] in done:t['state']='done'
   if t['id'] in running:t['state']='running'
  s['timeline']=ls(tasks,cursor=cursor)
 def add(i,title,ex,t=None):
  steps.append(step(i,title,ex,s,[metric('ready','Locally ready gradients',sum(x is not None for x in s['grads']['slots']),'gradients'),metric('reduced','Completed AllReduce results',sum(x is not None and str(x['value']).endswith('reduced') for x in s['grads']['slots']),'gradients'),metric('exposed','Completed final overhead and network tail',max(0,min(s['timeline']['cursor'],tail_end)-last_done),'schematic units')],f'{reduced} of 2 required reduced gradients are available; weight remains {s["weight"]["value"]}.',t))
 add('start','Begin the final two backward layers','Earlier layers L4 and L3 from the source diagram are outside this focused scene and already complete.')
 ready=1;s['grads']=ps(['g2 local',None],owner='Training step');ts(['b2'],['o2'],b);add('ready2','Finish backward L2','The local L2 gradient becomes available. Its collective has not launched.',tr('execute',['b2','grads'],'Complete one backward layer and expose its gradient.'))
 ts(['o2'],['a2','b1'],boundary);add('launch2','Finish L2’s nonoverlappable launch overhead','The GPU has paid the launch overhead; network L2 can run while GPU backward L1 runs.',tr('execute',['o2','a2','b1'],'Complete the launch overhead and hand off work to concurrent engines.'))
 ready=2;reduced=1;s['grads']=ps(['g2 reduced','g1 local'],owner='Training step');ts(['b1','a2'],['o1'],last_done);add('parallel','Finish the concurrent L1 backward and L2 transfer','The declared equal-length interval finishes both parallel tasks. L1 is now ready, but its AllReduce has not launched.',tr('execute',['b1','a2','grads'],'Complete the parallel backward/communication interval on separate engines.'))
 exposed=o;ts(['o1'],['a1','idle'],last_launch);add('launch1','Finish L1’s nonoverlappable launch overhead','There is no backward computation left to hide the final transfer. The optimizer still waits for L1.',tr('execute',['o1','a1','idle'],'Complete final launch overhead and start the exposed transfer.'))
 reduced=2;exposed=o+net;s['grads']=ps(['g2 reduced','g1 reduced'],owner='Training step',state='ready');ts(['a1','idle'],['opt'],tail_end);add('tail','Finish the exposed L1 AllReduce','All required reduced gradients are finally available; only now may the optimizer execute.',tr('execute',['a1','idle','grads'],'Complete the final network transfer on the critical path.'))
 s['weight']=rs(10-1,previous=10,expression='10 − 1');ts(['opt'],cursor=end);add('update','Complete the optimizer update','A declared toy optimizer decrement consumes the complete reduced gradient set.',tr('execute',['opt','weight'],'Commit one shared optimizer update after the last collective.'))
 return lesson('fig-comm-compute-overlap','Expose the final communication tail','Which transfer is hidden by backward work, and which still delays the update?','Focused final two layers L2 and L1 of the four-layer diagram; earlier layers are complete. Backward tasks and transfers each take three schematic units; each launch occupies the GPU for one. No measured speedup is claimed. The toy final weight decrement is one.','schedule',objects,steps,['A layer AllReduce starts only after that layer is ready and its GPU launch overhead completes.','L2 transfer overlaps L1 backward on separate engines.','The final L1 launch overhead and transfer have no later backward work to hide behind.','Optimizer state changes only after every required reduced gradient is available.'])

def retry():
 outputs=[2,4,6];grads=[1,2,3];initial_weight=10;from fractions import Fraction
 final_weight=initial_weight-sum(grads)*Fraction(1,6)
 objects=[lanes('maps','Retryable map/reduce tasks',0,['Map 1','Map 2','Map 3','Reducer'],7),pool('outputs','Retained map outputs',3,1,['M1','M2','M3']),lanes('training','One synchronous training step',2,['Worker 0','Worker 1','Worker 2','Collective / update'],7),pool('gradients','Contributions and collective result',4,3,['g0','g1','g2','sum']),reg('reduced','Reducer output',4),reg('weights','Shared weight version / value',5,'toy scalar','weight units')]
 mt=[seg('m1',0,0,2,'Map 1','running'),seg('m2',1,0,2,'Map 2','running'),seg('m3',2,0,1,'Attempt M3 (fails)','running'),seg('retry',2,1,5,'Retry only M3'),seg('reduce',3,5,6,'Dependent reducer')]
 tt=[seg('w0',0,0,2,'Gradient 0','running'),seg('wait0',0,2,5,'Wait for worker 2',role='control'),seg('w1',1,0,2,'Gradient 1','running'),seg('wait1',1,2,5,'Wait for worker 2',role='control'),seg('w2',2,0,5,'Late gradient 2','running'),seg('collective',3,5,6,'AllReduce',role='communication'),seg('update',3,6,7,'Shared update')]
 s={'maps':ls(mt),'outputs':ps([None]*3),'training':ls(tt),'gradients':ps([None]*4),'reduced':rs(None),'weights':rs(initial_weight)};steps=[]
 def task(tasks,ids,state):
  for t in tasks:
   if t['id'] in ids:t['state']=state
  s['maps']=ls(mt);s['training']=ls(tt)
 def add(i,title,ex,t=None):
  maps=sum(x is not None for x in s['outputs']['slots']);gs=sum(x is not None for x in s['gradients']['slots'][:3])
  steps.append(step(i,title,ex,s,[metric('mapsDone','Retained map outputs',maps,'outputs'),metric('gradientsReady','Required gradients ready',gs,'gradients'),metric('weight','Shared weight',s['weights']['value'],'weight units')],f'{maps} map outputs are retained; {gs} gradients are ready; shared weight is {s["weights"]["value"]}.',t))
 add('start','Establish the two dependency patterns','The panels are separate illustrative jobs. Their lanes show dependency ordering, not a shared clock across panels.')
 task(mt,['m3'],'blocked');task(mt,['retry'],'running');add('fail','Fail only map 3’s first attempt','The scheduler retries the failed task. Maps 1 and 2 continue toward completion without restarting.',tr('execute',['m3','retry'],'Record one map failure and activate only its retry.'))
 s['outputs']=ps(outputs[:2]+[None],owner='Map job');task(mt,['m1','m2'],'done');add('maps','Finish maps 1 and 2 independently','Their output records persist even though map 3 and the dependent reducer have not completed.',tr('execute',['m1','m2','outputs'],'Complete two independent map tasks in parallel.'))
 s['outputs']=ps(outputs,owner='Map job');task(mt,['retry'],'done');task(mt,['reduce'],'running');add('retry','Finish the retry of map 3','The reducer can now run because every map output it requires exists.',tr('execute',['retry','outputs','reduce'],'Complete the retried map and satisfy the reducer dependency.'))
 s['reduced']=rs(sum(outputs),expression=' + '.join(map(str,outputs)));task(mt,['reduce'],'done');add('reduce','Finish the dependent reducer','Local retry preserved other tasks’ progress, but the missing map output still delayed this reducer.',tr('execute',['reduce','reduced'],'Compute the reducer from all three retained records.'))
 s['gradients']=ps(grads[:2]+[None,None],owner='Training step');task(tt,['w0','w1'],'done');task(tt,['wait0','wait1'],'running');add('healthy','Finish gradients 0 and 1','Healthy workers retain their contributions and wait. The old weight remains in place while worker 2 is late.',tr('execute',['w0','w1','gradients'],'Complete the two healthy gradient tasks in parallel.'))
 s['gradients']=ps(grads+[None],owner='Training step');task(tt,['w2','wait0','wait1'],'done');task(tt,['collective'],'running');add('late','Finish the last required gradient','The complete contribution set now permits AllReduce; gradient readiness itself is not the shared parameter update.',tr('execute',['w2','gradients','collective'],'Complete the late gradient and release the collective dependency.'))
 s['gradients']=ps(grads+[sum(grads)],owner='Training step');task(tt,['collective'],'done');task(tt,['update'],'running');add('sum','Finish the shared AllReduce','The scalar collective returns the sum of all contributions. The old shared weight still remains until optimizer commit.',tr('execute',['collective','gradients'],'Complete the collective reduction of all required contributions.'))
 s['weights']=rs(int(final_weight),previous=initial_weight,expression=f'{initial_weight} − {sum(grads)} / 6');task(tt,['update'],'done');add('update','Commit the shared weight update','The toy optimizer uses learning rate one sixth. This update cannot omit the late worker’s gradient.',tr('execute',['update','weights'],'Complete the shared optimizer update after the collective.'))
 return lesson('fig-traditional-vs-ml-fleet','Compare task retry with a shared update','Which progress survives a local retry, and which waits for every required contribution?','Separate three-map and three-worker toy jobs. Schematic lane positions indicate dependencies; panel steps are pedagogical inspections, not synchronized wall-clock samples. Map outputs and scalar gradients are illustrative; optimizer learning rate is one sixth.','schedule',objects,steps,['A failed map does not invalidate completed output records from other maps.','A reducer still waits for every map output it depends on.','The synchronous shared update requires every worker contribution and the completed collective.','The shared weight remains unchanged while any required gradient is missing.'])

def pipeline():
 # The snapshot starts at an explicitly declared warm pipeline instant.
 # Durations retain one shared schematic axis; delayed read B3 causes the drain.
 read2=(0,2);prep2=(2,4);train1=(2,6);train2=(6,10);read3=(2,11);prep3=(11,13);train3=(13,17)
 objects=[lanes('timeline','Read, prepare, and train batch identities',0,['Storage','CPU','GPU'],17),pool('raw','Host batch being prepared',1,1,['Raw batch']),pool('queue','Prepared prefetch queue',1,2,['Ready batch']),obj('gpu','chamber','GPU current batch','compute',dict(slots=1),3),obj('done','sequence','Completed batches','output',dict(slots=3,slotLabels=['B1','B2','B3']),4)]
 tasks=[seg('r2',0,*read2,'Read B2','done','memory'),seg('r3',0,*read3,'Read B3 (delayed)','running','memory'),seg('p2',1,*prep2,'Prepare B2','running'),seg('p3',1,*prep3,'Prepare B3'),seg('t1',2,*train1,'Train B1','running'),seg('t2',2,*train2,'Train B2'),seg('wait',2,train2[1],prep3[1],'GPU input wait',role='control'),seg('t3',2,*train3,'Train B3')]
 s={'timeline':ls(tasks,cursor=prep2[0]),'raw':ps(['B2'],owner='Loader'),'queue':ps([None]),'gpu':{'values':['B1'],'status':'Warm pipeline: B1 already prepared'},'done':{'values':[None]*3,'status':'No batch completed in this scene'}};steps=[];wait=0
 def task(done=(),run=(),cursor=None):
  for t in tasks:
   if t['id'] in done:t['state']='done'
   if t['id'] in run:t['state']='running'
  s['timeline']=ls(tasks,cursor=cursor)
 def add(i,title,ex,t=None):
  occupied=sum(v is not None for v in s['queue']['slots']);finished=sum(v is not None for v in s['done']['values'])
  s['done']['status']=('Completed records retained: '+', '.join(str(v)for v in s['done']['values']if v is not None)) if finished else 'No batch completed in this scene'
  steps.append(step(i,title,ex,s,[metric('queued','Prepared queue occupancy',occupied,'batches'),metric('completed','Completed batches',finished,'batches'),metric('wait','Exposed input wait elapsed',wait,'schematic units')],f'{finished} batches completed; {occupied} prepared batch is queued; input wait elapsed is {wait} schematic units.',t))
 add('warm','Inspect three batch identities in flight','GPU trains B1, CPU prepares B2, and storage reads B3. Initial warm-up of B1 is outside this focused scene.')
 s['raw']=ps([None]);s['queue']=ps(['B2'],owner='Loader',state='ready');task(['p2'],cursor=prep2[1]);add('prepare2','Finish preparing B2 into the ready queue','The prepared batch waits in host memory while the GPU finishes B1; overlap does not merge batch identities.',tr('transform',['raw','queue'],'Prepare one raw batch and retain its ready representation.',{'object':'raw','index':0},{'object':'queue','index':0}))
 s['gpu']={'values':[None],'status':'B1 completed'};s['done']['values'][0]='B1';task(['t1'],cursor=train1[1]);add('finish1','Finish GPU training on B1','B1’s completed record remains visible. The already prepared B2 is ready for dispatch.',tr('execute',['t1','gpu','done'],'Complete GPU work on batch B1.'))
 s['queue']=ps([None]);s['gpu']={'values':['B2'],'status':'B2 dispatched from the ready queue'};task(run=['t2'],cursor=train2[0]);add('dispatch','Dispatch the queued B2 to the GPU','Dispatch consumes the one queued prepared batch. Storage is still reading delayed B3.',tr('move',['queue','gpu'],'Move the prepared batch to its GPU consumer.',{'object':'queue','index':0},{'object':'gpu','index':0}))
 s['gpu']={'values':[None],'status':'No prepared B3; GPU waiting'};s['done']['values'][1]='B2';task(['t2'],['wait'],train2[1]);add('drain','Finish B2 and expose an empty input queue','The prefetch cushion is exhausted before B3 arrives. No useful GPU computation can continue.',tr('execute',['t2','gpu','done','wait'],'Complete GPU work on B2; the next required batch is absent.'))
 wait=read3[1]-train2[1];s['raw']=ps(['B3'],owner='Loader');task(['r3'],['p3'],read3[1]);add('read3','Finish delayed storage delivery of B3','B3 is raw data, so arrival alone cannot resume the GPU. CPU preparation remains a dependency.',tr('write',['raw','r3'],'Complete one read and deliver its raw batch.',to={'object':'raw','index':0}))
 wait=prep3[1]-train2[1];s['raw']=ps([None]);s['queue']=ps(['B3'],owner='Loader',state='ready');s['gpu']['status']='Prepared B3 awaits dispatch';task(['p3','wait'],cursor=prep3[1]);add('prepare3','Finish preparing B3','The newly ready batch ends exposed input wait, but dispatch remains an explicit next action.',tr('transform',['raw','queue'],'Finish CPU preparation of the required next batch.',{'object':'raw','index':0},{'object':'queue','index':0}))
 s['queue']=ps([None]);s['gpu']={'values':['B3'],'status':'GPU resumes useful training'};task(run=['t3'],cursor=train3[0]);add('resume','Dispatch B3 and resume GPU training','Prefetching hid earlier delivery work, but the delayed read and preparation exposed a real gap once the queue drained.',tr('move',['queue','gpu'],'Move B3 from ready storage into its GPU consumer.',{'object':'queue','index':0},{'object':'gpu','index':0}))
 return lesson('fig-data-pipeline-vol2','Watch the prefetch queue drain','When does overlap stop hiding storage and preparation latency?','Focused warm pipeline with three batches B1–B3 and one prepared queue slot. B1 warm-up is omitted. Shared schematic intervals explicitly model one delayed B3 read; they are a supplementary variance case rather than the static figure’s four-batch throughput comparison.','schedule',objects,steps,['The GPU consumes only prepared batches, never raw read output.','Batch identity persists through read, preparation, queueing, and GPU consumption.','Queue dispatch removes a prepared batch from the queue.','An empty queue with unfinished upstream work exposes GPU input wait.'],checks=[dict(kind='metricEqualsOccupied',metric='queued',object='queue'),dict(kind='metricEqualsNonNull',metric='completed',object='done')])

def build():
 data=[retry(),gpu(),pfc(),pipeline(),checkpoint(),overlap(),hierarchy(),ring()]
 results={
 'fig-traditional-vs-ml-fleet':{
  'start':'The map job retains independent records; the training job must eventually commit one shared update.',
  'maps':'Maps 1 and 2 have durable-in-this-scene output records; map 3 cannot erase their completed work.',
  'fail':'Only the failed attempt is retried. Other maps continue; no completed task is invalidated.',
  'retry':'The missing record now exists, so the dependent reducer becomes runnable.',
  'reduce':'The reducer finishes from the retained records; independent map completion did not remove its dependencies.',
  'healthy':'Healthy workers have finished local work, but the missing third contribution holds the shared weight unchanged.',
  'late':'All contributions now exist; the collective may run, while the old shared weight is still retained.',
  'sum':'The complete collective result is ready; the optimizer commit remains a separate dependency.',
  'update':'Every required contribution is included in the new shared weight.'},
 'fig-gpudirect-data-path':{
  'source':'The source tensor is retained in GPU memory; no host bounce or fabric transfer has occurred.',
  'stage':'A host copy now exists alongside the original GPU tensor; this host write is the first bounce cost.',
  'read':'The NIC has read the retained host copy; host traffic includes both a payload write and a payload read.',
  'peer-read':'The NIC has a copy directly from GPU memory. The host buffer remains empty.',
  'send':'The full tensor still traverses the fabric; avoiding the host bounce does not remove network payload.',
  'release':'The completed send no longer needs its NIC slot. The GPU source tensor remains available.'},
 'fig-pfc-credit-deadlock':{
  'open':'One queue still has downstream space; the occupied dependency cycle has not closed.',
  'close':'Every queue is occupied by a packet needing the next full queue; none has a free destination.',
  'pause':'The clockwise waiting cycle and counterclockwise PAUSE messages hold every resident packet.',
  'drop':'The watchdog opens one slot by losing a packet. That loss is a recovery cost, not delivery.',
  'resume':'Queue b4 may send into b1 because both downstream space and PAUSE clearance now exist.',
  'forward':'P4 advances into b1 and frees b4. Breaking one dependency permits forward progress.'},
 'fig-data-pipeline-vol2':{
  'warm':'Three different batches occupy three stages concurrently; the GPU still needs a prepared successor.',
  'prepare2':'B2 waits ready in the prefetch queue while B1 continues on the GPU.',
  'finish1':'B1 is complete and B2 remains queued, so the GPU has a prepared successor.',
  'dispatch':'B2 runs on the GPU, leaving no prepared batch in reserve for the delayed B3 read.',
  'drain':'B2 is complete, but B3 is still upstream. Empty prefetch capacity cannot supply useful GPU work.',
  'read3':'B3 has arrived as raw data. The GPU must still wait for CPU preparation.',
  'prepare3':'B3 is finally prepared; the exposed input gap ends when the GPU dispatches this ready batch.',
  'resume':'The GPU resumes with B3. Prefetching hid earlier delivery work only until its reserve drained.'},
 'fig-storage-checkpoint-staging':{
  'cut':'The consistent cut is paused, and g0 remains the sole committed restore point.',
  'stage':'Immutable g1 shards exist locally. Resuming training does not make this generation restorable.',
  'train':'Live weight advances while staged shards retain the earlier cut; recovery still uses g0.',
  'copy0':'Only one g1 shard is durable. Publishing this incomplete generation would expose a broken restore point.',
  'copy1':'Both g1 shards are durable, but verification and manifest publication still precede commitment.',
  'verify':'The full shard set is verified. The restore pointer remains g0 until the atomic manifest commit.',
  'publish':'The committed manifest now names complete g1, so the restore point can advance safely.',
  'release':'Local staging is free only after durable copies and publication have finished using it.'},
 'fig-comm-compute-overlap':{
  'start':'L2 backward is still producing its gradient; neither final-layer collective can run yet.',
  'ready2':'L2 is locally ready, but launching its transfer still occupies the GPU.',
  'launch2':'Network L2 and GPU backward L1 now use separate engines concurrently.',
  'parallel':'L2 communication finishes beneath L1 backward. L1’s own transfer is still outstanding.',
  'launch1':'L1 transfer now runs with no later backward work to hide it; the optimizer must wait.',
  'tail':'The exposed final transfer is complete. All required reduced gradients can now feed the optimizer.',
  'update':'The shared optimizer state changes only after the residual communication tail has finished.'},
 'fig-hierarchical-allreduce':{
  'inputs':'Each GPU starts with its own contribution to both indexed shards.',
  'local':'Each GPU now owns one node-partial shard; the other shard belongs to its local peer.',
  'global':'Corresponding inter-node peers hold finished sums, but each GPU still lacks its complementary shard.',
  'gather':'Local copies distribute the complementary finished shard; every GPU now has the full result.'}}
 for item in data:
  key=item['key'].split(':',1)[1]
  for case in [item]+item.get('variants',[]):
   for st in case['steps']:
    sid=st['id']
    if key in results:st['snapshot']['result']=results[key][sid]
    elif sid=='inputs':st['snapshot']['result']='Each chunk contains only its local rank’s contribution; no global sum is complete.'
    elif sid.startswith('reduce'):
     st['snapshot']['result']=('Each GPU now owns one complete sum at a different shard index; other resident tiles are still partial.' if sid=='reduce3' else 'The received partial adds previously absent ranks to the same chunk index; brackets expose exactly which contributions are included.')
    else:st['snapshot']['result']=('Every GPU now holds every finished sum. AllGather copied the sums without adding contributions again.' if sid=='gather3' else 'Finished sums propagate clockwise and replace stale partials; already finished source chunks remain retained.')
 return data

if __name__=='__main__':
 import sys
 data=build()
 target=OUT/'vol2-a.json'
 if '--write' in sys.argv:
  OUT.mkdir(parents=True,exist_ok=True)
  target.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
 else:
  actual=json.loads(target.read_text())
  assert actual==data,'Authored JSON differs from computed declared-input snapshots'
  for item in data:
   for case in [item]+item.get('variants',[]):
    case_id=case.get('id','original')
    for state in case['steps']:
     print(item['key']+'/'+case_id+'/'+state['id']+': '+', '.join(f'{m["id"]}={m["value"]} {m["unit"]}'.rstrip() for m in state['snapshot']['metrics']))

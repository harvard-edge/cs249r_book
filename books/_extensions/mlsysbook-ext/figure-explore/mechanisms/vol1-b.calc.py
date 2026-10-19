#!/usr/bin/env python3
"""Recompute declared toy lessons; default checks JSON without writing it."""
import copy,json,math,pathlib,sys

def obj(i,k,l,role,c,row,col=0,span=1):
 return dict(id=i,kind=k,label=l,role=role,config=c,layout={'wide':dict(row=row,column=col,columnSpan=span),'narrow':dict(row=row*2+col,column=0,columnSpan=1)})
def seq(i,l,n,row,col=0,role='input'):return obj(i,'sequence',l,role,{'slots':n},row,col)
def pool(i,l,n,row,col=0,role='memory'):return obj(i,'pool',l,role,{'slots':n,'columns':min(n,4)},row,col)
def reg(i,l,row,col=0,typ='scalar',unit=''):return obj(i,'register',l,'compute',{'type':typ,'unit':unit},row,col)
def mat(i,l,r,c,row,col=0,role='input'):return obj(i,'matrix',l,role,{'rows':r,'columns':c},row,col)
def S(v,status=''):return {'values':v,'status':status}
def R(v,status='',expr=None,prev=None):
 d={'value':v,'status':status}
 if expr is not None:d['expression']=expr
 if prev is not None:d['previous']=prev
 return d
def P(v,status=''):return {'slots':v,'status':status}
def cell(v,own='training step',status='retained'):return dict(value=v,owner=own,status=status)
def metric(i,l,v,u):return dict(id=i,label=l,value=v,unit=u)
def step(i,title,ex,states,result,kind=None,subjects=None,frm=None,to=None,metrics=None,tokens=None):
 tr=None if kind is None else dict(kind=kind,subjects=subjects or [],meaning=ex)
 if tr is not None:
  if frm:tr['from']=frm
  if to:tr['to']=to
 return dict(id=i,title=title,explanation=ex,snapshot=dict(objects=copy.deepcopy(states),tokens=tokens or [],metrics=metrics or [],activeConnections=[],result=result),transition=tr)
def lesson(key,title,q,scope,fam,objects,steps,invs,checks=None):
 d=dict(schema=1,key='vol1:'+key,title=title,question=q,scope=scope,family=fam,objects=objects,steps=steps,invariants=invs)
 if checks:d['checks']=checks
 return d
def anchor(o,i=None):return {'object':o,**({} if i is None else {'index':i})}
def cp(i,v,source,dest,own='training step'):
 return dict(id=i,label=str(v),value=v,role='compute',owner=own,at=dest,state='copy',sourceAt=source)

def checkpoint():
 # A completed forward segment a0 -> a1 -> a2; a0 and a2 are checkpoints.
 O=[seq('network','Forward dependency: a0 → a1 → a2',3,0),pool('saved','Saved checkpoints',2,1),pool('scratch','Reconstructed activation',1,1,1),seq('backward','Backward consumers: a2 then a1',2,2,role='output')]
 st={'network':S(['a0','a1','a2']),'saved':P([cell('a0'),cell('a2')]),'scratch':P([None]),'backward':S([None,None])}; out=[]; re=0
 def add(i,t,e,res,kind=None,sub=None):
  occupied=sum(x is not None for o in ['saved','scratch'] for x in st[o]['slots'])
  out.append(step(i,t,e,st,res,kind,sub,metrics=[metric('live','Live activation slots',occupied,'slots'),metric('recomputed','Extra forward evaluations',re,'evaluations')]))
 add('saved','Start after the forward segment','Only selected boundaries were retained; a1 is absent.','The backward pass will need an activation that is no longer resident.')
 st['scratch']=P([cell('a1',status='ready')],'Recomputed from a0');re+=1
 add('recompute','Recompute a1 from retained a0','Evaluate the missing internal activation from the saved boundary. The read leaves a0 intact.','Reconstructed a1 occupies transient memory until its backward consumer finishes.','execute',['scratch'])
 st['backward']['values'][0]='done: ∇a1'
 add('backward2','Execute backward for layer 2','Layer 2 reads a1 and the retained output boundary a2 to produce its gradient.','a1 remains live because the next backward operation still consumes it.','execute',['backward'])
 st['saved']['slots'][1]=None
 add('release2','Release checkpoint a2','Its final backward consumer has finished.','Releasing a2 frees a slot; a0 and a1 remain resident.','release',['saved'])
 st['backward']['values'][1]='done: ∇a0'
 add('backward1','Execute backward for layer 1','The final operation in this segment consumes a0 and reconstructed a1.','Both segment backward operations have finished.','execute',['backward'])
 st['scratch']['slots'][0]=None
 add('release1','Release reconstructed a1','No backward consumer in this segment needs a1 again.','The temporary slot is free; the earlier checkpoint a0 remains for upstream work.','release',['scratch'])
 return lesson('fig-activation-checkpointing','Saved boundaries and transient reconstruction','When can a reconstructed activation be freed?','Toy two-layer segment after forward completion. Activation slots have equal schematic size; weights, gradients, RNG replay, and other segments are outside this view. a0 remains needed upstream.','ownership',O,out,['A checkpoint read does not remove its saved activation.','Reconstructed a1 remains resident through its final backward consumer.','The toy counts show this segment’s activation lifetime, not a universal memory-saving ratio.'])

def accumulation():
 sizes=[2,2,2];mean_grads=[3,-1,4];total=sum(sizes);weighted=[n*g/total for n,g in zip(sizes,mean_grads)];w=10.;lr=.3
 O=[seq('batches','Micro-batch mean gradients at fixed parameter',3,0),pool('acts','Current micro-batch activations',1,1),reg('sum','Normalized gradient accumulator',1,1),reg('weight','Parameter w',2),reg('updates','Optimizer updates',2,1,'count','updates')]
 st={'batches':S(mean_grads),'acts':P([None]),'sum':R(0.),'weight':R(w),'updates':R(0)}; out=[]; completed=0
 def add(i,t,e,res,kind=None,sub=None):out.append(step(i,t,e,st,res,kind,sub,metrics=[metric('examples','Examples accumulated',completed,'examples'),metric('live','Live micro-batches',sum(x is not None for x in st['acts']['slots']),'micro-batches')]))
 add('start','Start one logical batch','Each micro-batch has equal size. Its mean gradient is weighted by its share of the logical batch.','The parameter stays fixed until all contributions are accumulated.')
 for j,(n,g,c) in enumerate(zip(sizes,mean_grads,weighted)):
  st['acts']=P([cell('batch '+str(j+1))],'Forward activations ready')
  add('forward'+str(j+1),'Run forward for micro-batch '+str(j+1),'Only this micro-batch needs live forward activations.','The saved activations are ready for its backward pass.','execute',['acts'])
  old=st['sum']['value'];st['sum']=R(old+c,'Backward contribution retained',f'{old:g} + ({n}/{total}) × {g:g}',old);st['acts']=P([None],'Backward consumed activations');completed+=n
  add('backward'+str(j+1),'Backpropagate and accumulate micro-batch '+str(j+1),'Backward adds the normalized contribution and consumes this micro-batch’s saved activations.','The gradient buffer persists while the next micro-batch reuses the activation slot.','execute',['sum','acts'])
 st['weight']=R(w-lr*sum(weighted),'One SGD update',f'{w:g} − {lr:g} × {sum(weighted):g}',w);st['updates']=R(1)
 add('update','Apply one SGD update','The complete logical-batch gradient now drives exactly one parameter update.','The parameter changes once, after every micro-batch has contributed.','execute',['weight','updates'])
 st['sum']=R(0.,'Ready for the next logical batch');completed=0
 add('clear','Clear the accumulation buffer','Reset only after the update has consumed the complete gradient.','The next logical batch starts with an empty gradient accumulator.','write',['sum'])
 return lesson('fig-grad-accumulation','Accumulate gradients before updating parameters','What survives between micro-batches, and when does the parameter change?','Three equal toy micro-batches of two examples, with mean gradients 3, −1, and 4 evaluated at w = 10. SGD uses learning rate 0.3. Fixed gradients isolate scheduling; batch-dependent layers and stochastic equivalence are outside the toy.','reduction',O,out,['The parameter remains fixed during all micro-batch forward/backward operations.','Mean gradients are weighted by micro-batch example count divided by total examples.','Backward may consume and release its activations while retaining the accumulated gradient.','The optimizer update and buffer reset are separate actions.'],[{'kind':'constantValues','object':'batches'}])

def prefetch():
 # Declared toy durations; resource constraints are CPU producer and GPU consumer.
 read=2;train=3;n=2
 schedules={
 'overlap':[(0,0,read,'Fill A','input'),(1,read,read+train,'Train A','compute'),(0,read,read*2,'Fill B','input'),(1,read+train,read+train*2,'Train B','compute')],
 'serial':[(0,0,read,'Fill A','input'),(1,read,read+train,'Train A','compute'),(0,read+train,read*2+train,'Fill B','input'),(1,read*2+train,read*2+train*2,'Train B','compute')]}
 horizon=max(max(s[2] for s in v) for v in schedules.values())
 for ts in schedules.values():
  for lane in range(2):
   intervals=sorted((a,b) for ln,a,b,_,_ in ts if ln==lane)
   assert all(a[1]<=b[0] for a,b in zip(intervals,intervals[1:]))
  assert ts[0][2]<=ts[1][1] and ts[2][2]<=ts[3][1]
 O=[obj('timeline','lanes','Producer and consumer schedule','compute',{'lanes':['Host open/read','GPU train'],'start':0,'end':horizon,'axisLabel':'Toy time units','ticks':list(range(0,horizon+1,2))},0,span=2),pool('buffers','Two bounded batch buffers',2,1),seq('done','Finished training batches',2,1,1,'output')]
 def build(mode):
  ts=schedules[mode];end=max(t[2] for t in ts);st={'timeline':{'segments':[],'cursor':0,'status':'No work completed'},'buffers':P([None,None]),'done':S([None,None])};out=[]
  # Snapshots at causal task-completion events; buffer owns data until training finishes.
  events=sorted(range(len(ts)),key=lambda j:ts[j][2]);events=[None]+events
  finished=set()
  for k,event in enumerate(events):
   if event is not None:finished.add(event)
   now=0 if event is None else ts[event][2]
   seg=[]
   for j,(lane,a,b,label,role) in enumerate(ts):seg.append(dict(id='task'+str(j),lane=lane,start=a,end=b,label=label,state='done' if j in finished else ('running' if a<=now<b else 'waiting'),role=role))
   st['timeline']={'segments':seg,'cursor':now,'status':'One resource per lane'}
   for batch,fill,consume in [('A',0,1),('B',2,3)]:
    idx=ord(batch)-ord('A')
    st['buffers']['slots'][idx]=(cell('filling '+batch,'batch '+batch,'inflight') if ts[fill][1]<=now<ts[fill][2] else (cell('batch '+batch,'batch '+batch,'inflight' if ts[consume][1]<=now<ts[consume][2] else 'ready') if fill in finished and consume not in finished else None))
    st['done']['values'][idx]=batch if consume in finished else None
   completed=sum(x is not None for x in st['done']['values'])
   if event is None:t='Begin filling the first reserved buffer';e='Each batch must finish host open/read before its GPU training can begin.';r='Two slots bound prefetch capacity; a GPU consumer keeps its buffer occupied.'
   else:
    t='Complete '+ts[event][3];e='The producer fills a separate buffer.' if ts[event][0]==0 else 'Training consumes this batch and releases its buffer only at completion.';r='The next batch may become ready while GPU training still uses the other slot.' if event==2 and mode=='overlap' else ('All training is complete.' if completed==n else 'Data ownership and task readiness agree at this boundary.')
   out.append(step('event'+str(k),t,e,st,r,None if event is None else 'execute',[] if event is None else ['task'+str(event),'buffers','done'],metrics=[metric('elapsed','Elapsed toy time',now,'time units'),metric('makespan','Schedule makespan',end,'time units'),metric('finished','Trained batches',completed,'batches')]))
  return out
 d=lesson('fig-fetching-optimized','Prefetch into a separate buffer','How can host work overlap GPU training without overwriting its input?','Toy two-batch pipeline: host open/read takes two time units per batch and GPU training takes three. This isolates the finite-buffer dependency; it does not reproduce the printed minute-scale schedule or measured hardware timing.','schedule',O,build('overlap'),['Every batch finishes host production before GPU training begins.','The producer never overwrites a buffer still owned by GPU training.','Host and GPU each run at most one task at a time; their tasks may overlap.','The serial comparison uses identical stage durations and batch count.'])
 d['variants']=[dict(id='serial',label='Serial producer and GPU',scope='Same two toy batches, stage durations, and buffers; host work waits for each preceding GPU task.',steps=build('serial'))]
 return d

def attention():
 # One scalar query and two one-key blocks; score = q*k for d = 1.
 query=1.;keys=[0.,math.log(3.)];scores=[query*k for k in keys];values=[2.,6.];m=-math.inf;l=0.;numer=0.
 O=[seq('inputs','HBM inputs: q, k1, v1, k2, v2',5,0),pool('hbm','HBM output slot',1,0,1),pool('tile','SRAM current block',2,1),seq('running','SRAM state: max, normalizer, numerator',3,1,1,'compute'),reg('output','Normalized output ready to write',2)]
 st={'inputs':S([query,round(keys[0],6),values[0],round(keys[1],6),values[1]],'Query, keys, and values remain resident'),'hbm':P([None]),'tile':P([None,None]),'running':S([None,0.,0.]),'output':R(None)};out=[]
 def add(i,t,e,r,kind=None,sub=None,tokens=None):out.append(step(i,t,e,st,r,kind,sub,metrics=[metric('hbmIntermediate','HBM score/probability writes',0,'elements'),metric('sram','Resident SRAM block slots',sum(x is not None for x in st['tile']['slots']),'slots')],tokens=tokens))
 add('start','Begin a row of tiled attention','The HBM inputs stay resident; the output slot is unwritten.','Only current blocks and running state will be retained on chip.')
 for j,(score,v) in enumerate(zip(scores,values)):
  st['tile']=P([cell(round(keys[j],6),'attention row','ready'),cell(v,'attention row','ready')],'Current key and value block')
  add('load'+str(j),'Load block '+str(j+1)+' into SRAM','Read the next key/value block without removing the HBM source.','The current block is available for an online softmax update.','copy',['inputs'],[cp('key'+str(j),round(keys[j],6),anchor('inputs',j*2+1),anchor('tile',0),'attention row'),cp('value'+str(j),v,anchor('inputs',j*2+2),anchor('tile',1),'attention row')])
  new_m=max(m,score);alpha=0 if m==-math.inf else math.exp(m-new_m);p=math.exp(score-new_m);l=alpha*l+p;numer=alpha*numer+p*v;m=new_m
  st['running']=S([round(m,6),round(l,6),round(numer,6)],'Rescaled to the current maximum')
  add('merge'+str(j),'Merge block '+str(j+1)+' into running state','The block kernel computes q × k and merges its weighted value. A higher maximum rescales the retained normalizer and numerator.','The running state now represents every block seen so far, without an HBM probability array.','execute',['running'])
  st['tile']=P([None,None],'Block fully consumed')
  add('release'+str(j),'Release current block storage','The running state contains the information needed from this block.','The SRAM block slots can be reused; accumulated state remains.','release',['tile'])
 assert math.isclose(numer/l,sum(math.exp(sc)*v for sc,v in zip(scores,values))/sum(math.exp(sc) for sc in scores))
 st['output']=R(round(numer/l,6),'Complete row normalized',f'{round(numer,6)} / {round(l,6)}')
 add('normalize','Normalize the completed row','Divide only after all key/value blocks have contributed.','The complete attention output is ready; it equals the materialized softmax result.','execute',['output'])
 st['hbm']=P([cell(round(numer/l,6),'attention row','ready')],'Completed output written')
 add('write','Write the completed output to HBM','Inputs and final outputs still occupy HBM.','No full score or probability matrix crossed into HBM in this toy forward path.','write',['hbm'])
 return lesson('fig-training-attention-storage','Keep the softmax working set in SRAM','What must survive when attention moves to the next block?','Toy forward attention row: scalar query 1, scalar keys 0 and ln(3), values 2 and 6, head dimension 1. Each SRAM key/value block feeds one online-softmax kernel update. Six-decimal display rounding does not change the full-precision calculation. No training-memory total or hardware capacity is claimed.','ownership',O,out,['HBM input values remain unchanged during block reads.','The normalizer and numerator are expressed relative to the current running maximum.','A block is released only after its contribution enters running state.','Completed output is written to HBM; score and probability intermediates are not.'],[{'kind':'constantValues','object':'inputs'}])

def selection():
 # Counts are a toy cohort; this never duplicates the chapter hardware registry.
 images=list('ABCDEF');keep={'A','C','E'}
 O=[seq('storage','Storage image records',len(images),0),seq('host','Host decoded images',len(images),1),seq('link','PCIe transfers',len(images),1,1),seq('gpu','GPU training examples',len(images),2,role='output')]
 def build(mode):
  selected=[x for x in images if x in keep];up=selected if mode=='materialized' else images
  st={'storage':S(up+[None]*(len(images)-len(up))),'host':S([None]*len(images)),'link':S([None]*len(images)),'gpu':S([None]*len(images))};out=[]
  for k,target in enumerate([None,'host','link','gpu']):
   if target:
    records=selected if target=='gpu' else up
    # Stream filtering occurs on the device, after the upstream pipeline.
    st[target]=S(records+[None]*(len(images)-len(records)))
   metrics=[metric(i,'Examples at '+i,sum(x is not None for x in st[i]['values']),'examples') for i in ['storage','host','link','gpu']]
   title=['Start the declared epoch','Decode the stored records','Transfer this epoch’s records','Train on the selected records'][k]
   ex=['The fixed selection keeps A, C, and E.','The host must decode every record it reads before an after-decode filter can discard it.','Device-side stream filtering has not acted yet; every decoded record crosses PCIe.' if mode=='stream' else 'Only the coreset records exist in this epoch and cross PCIe.','Only retained examples enter the training computation.'][k]
   result=['The case label fixes where selection acts.','Decoded work depends on the selection boundary.','The stream case retains transfer work that materialization removes.','A reduced GPU batch does not establish reduced storage or decode work.'][k]
   out.append(step('stage'+str(k),title,ex,st,result,None if k==0 else 'execute',[] if k==0 else [target],metrics=metrics))
  return out
 d=lesson('fig-data-selection-resnet-data-path','Selection saves work only downstream','Which stages still process images that training rejects?','Six-image toy cohort illustrates selection placement in the printed ResNet-50 data path. A, C, and E are retained. Stream filtering is explicitly placed on the device, after PCIe, matching the printed arithmetic-only saving. Counts are examples, not the figure’s H100 time, energy, or byte budgets. Coreset creation and scoring are excluded.','sequence',O,build('stream'),['The same three examples enter GPU training in both cases.','A device-side decision cannot recover storage, decoding, or PCIe work already performed.','A pre-materialized coreset removes rejected records before this epoch begins.'])
 d['variants']=[dict(id='materialized',label='Pre-materialized coreset',scope='Same retained examples; selection happened before this epoch, so rejected records are absent from storage. Coreset creation cost is outside the epoch.',steps=build('materialized'))]
 return d

def block_sparse():
 # 2 block rows × 2 block columns, 2×2 dense blocks: teaching shape, no hardware support claim.
 A=[[1,2,0,0],[3,4,0,0],[0,0,5,6],[0,0,7,8]];x=[1,2,3,4];coords=[(0,0),(1,1)];blocks=[[A[br*2+r][bc*2+c] for r in range(2) for c in range(2)] for br,bc in coords];dense=[sum(a*b for a,b in zip(row,x)) for row in A]
 O=[mat('weights','Logical sparse matrix (zero blocks remain zero)',4,4,0),seq('indices','Stored block coordinates',2,0,1),mat('packed','Packed nonzero 2×2 blocks',2,4,1),seq('input','Dense input vector x',4,1,1),seq('gather','Selected input block',2,2),seq('output','Retained output vector y',4,2,1,'output')]
 st={'weights':S(sum(A,[])),'indices':S([f'({r},{c})' for r,c in coords]),'packed':S(blocks[0]+blocks[1]),'input':S(x),'gather':S([None,None]),'output':S([0]*len(A))};out=[];mult=0
 def add(i,t,e,r,kind=None,sub=None,tokens=None):out.append(step(i,t,e,st,r,kind,sub,metrics=[metric('products','Executed scalar products',mult,'products'),metric('valueBytes','Packed value payload',sum(len(b) for b in blocks)*2,'bytes'),metric('indexBytes','Coordinate metadata',len(coords)*2*2,'bytes')],tokens=tokens))
 add('start','Start with zeroed output','Each coordinate names one stored dense block. Other logical blocks are zero.','Metadata overhead is additional to packed value storage.')
 for j,((br,bc),b) in enumerate(zip(coords,blocks)):
  st['packed']['window']={'row':j,'column':0,'rows':1,'columns':4}
  st['weights']['window']={'row':br*2,'column':bc*2,'rows':2,'columns':2}
  add('index'+str(j),'Decode block coordinate '+str(j+1),'The coordinate selects both the logical output rows and matching input columns.','The matching input block is determined by its column coordinate.','window',['packed','weights'])
  xs=x[bc*2:bc*2+2];st['gather']=S(xs,'Read without changing x')
  add('gather'+str(j),'Gather the matching input block','Read only the dense vector positions named by the block column.','The source vector remains intact for other block products.','copy',['input'],[cp('x'+str(j)+str(c),v,anchor('input',bc*2+c),anchor('gather',c)) for c,v in enumerate(xs)])
  for r in range(2):st['output']['values'][br*2+r]+=sum(b[r*2+c]*xs[c] for c in range(2))
  mult+=len(b)
  add('compute'+str(j),'Multiply the dense block and accumulate','One supported block operation updates its retained output rows.','Empty off-diagonal blocks are absent from the metadata and execute no block operation.','execute',['output'])
 assert st['output']['values']==dense
 return lesson('fig-block-sparse-gemm','Indices make sparse blocks executable','How does a stored block find its matching dense operand?','Toy 4×4 matrix with two 2×2 nonzero blocks, not the print figure’s illustrative 3×3 tiles. Packed values use two bytes and each explicit row/column coordinate uses two bytes. This is a teaching format, not a library ABI or latency claim.','spatial',O,out,['Sparse and dense execution compute the same already-pruned matrix-vector product.','A block column selects matching input elements; a block row selects output elements.','Unstored blocks are logical zeros, not uninitialized output.','Value bytes and metadata bytes are counted separately.'],[{'kind':'constantValues','object':'input'},{'kind':'constantValues','object':'weights'}])

def calibration():
 # Same 4 codes, integer affine zero point, fixed observations. Symmetric uses three of four codes.
 samples=[-.2,.3,.8,1.3];codes=[-2,-1,0,1]
 O=[obj('observed','numberLine','Fixed calibration samples','input',{'min':-2,'max':2,'ticks':[-2,-1,0,1,2],'axisLabel':'Real value'},0),obj('mapped','numberLine','Representable reconstruction levels','output',{'min':-2,'max':2,'ticks':[-2,-1,0,1,2],'axisLabel':'Real value'},1),seq('codes','Stored integer codes',4,2),reg('error','Mean squared reconstruction error',2,1)]
 cases=[('symmetric','Symmetric: scale 1.5, zero point 0',1.5,0,-1,1),('asymmetric','Asymmetric: scale 0.75, zero point −1',.75,-1,min(codes),max(codes)),('narrow','Narrow range: scale 0.25, zero point −1',.25,-1,min(codes),max(codes))]
 st={'observed':{'levels':[],'markers':[dict(id='sample'+str(i),value=v,label=str(v),role='input') for i,v in enumerate(samples)],'status':'Same observations in every mapping'},'mapped':{'levels':[],'markers':[],'status':'No quantizer applied'},'codes':S([None]*len(samples)),'error':R(None)};out=[]
 def add(i,t,e,r,kind=None,sub=None,clip=0,used=0):out.append(step(i,t,e,st,r,kind,sub,metrics=[metric('clipped','Clipped samples',clip,'samples'),metric('used','Reconstruction levels available',used,'levels')]))
 add('start','Fix samples and a four-code budget','All three mappings use the same observations and integer code storage.','Changing calibration changes represented real values, not the source samples.')
 for name,title,s,z,lo,hi in cases:
  assert lo in codes and hi in codes
  if name=='symmetric':assert z==0 and lo==-hi
  levels=[s*(q-z) for q in range(lo,hi+1)];qs=[max(lo,min(hi,math.floor(v/s+z+.5))) for v in samples];recon=[s*(q-z) for q in qs];clipped=sum(v<min(levels) or v>max(levels) for v in samples);mse=sum((a-b)**2 for a,b in zip(samples,recon))/len(samples)
  assert all(q in codes for q in qs)
  assert s*(z-z)==0
  st['mapped']={'levels':levels,'markers':[dict(id='recon'+str(i),value=v,label=f'x{i+1} → {v:g}',role='output') for i,v in enumerate(recon)],'status':title};st['codes']=S(qs);st['error']=R(round(mse,6))
  add(name,title,'Apply the declared affine map q = clamp(round(x/scale) + zero point). Original samples remain fixed.','The narrower grid improves spacing but clips the high samples.' if name=='narrow' else ('The skewed range assigns more usable levels to these observations.' if name=='asymmetric' else 'Real zero maps to integer zero; the most negative code is reserved to retain equal-magnitude endpoints.'),'transform',['mapped','codes','error'],clipped,len(levels))
 return lesson('fig-calibration-ranges','Range selection allocates finite codes','How does a narrower calibration range trade rounding for clipping?','Four fixed toy samples share signed codes −2, −1, 0, and 1. Symmetric mapping uses −1, 0, and 1 with zero point 0; code −2 is reserved for equal-magnitude endpoints. Asymmetric mappings use all four codes with zero point −1. This toy code width differs from printed INT8 and does not measure model accuracy.','representation',O,out,['The calibration observations remain unchanged in every state.','Each active level reconstructs as scale × (code − zero point).','Clipping counts source samples outside the represented endpoint interval.','Reconstruction error on these samples does not establish model accuracy.'])

def sparsity24():
 weights=[2,0,0,3,0,4,5,0];x=[1,2,3,4,5,6,7,8];groups=[weights[i:i+4] for i in range(0,len(weights),4)];positions=[[j for j,v in enumerate(g) if v!=0] for g in groups];packed=[[g[j] for j in p] for g,p in zip(groups,positions)];dense=sum(a*b for a,b in zip(weights,x))
 O=[mat('weights','Already-pruned weight groups',len(groups),len(groups[0]),0),mat('x','Dense activation groups',len(groups),len(groups[0]),0,1),seq('packed','Compressed retained values',4,1),seq('meta','Position tags inside each group',4,1,1),seq('gather','Metadata-selected activations',4,2),reg('sum','Retained INT32 dot product',2,1,'INT32','integer units')]
 for o in O[:2]:o['config'].update(rowLabels=['Group '+str(i+1) for i in range(len(groups))],columnLabels=[str(i) for i in range(len(groups[0]))])
 st={'weights':S(weights),'x':S(x),'packed':S([None]*4),'meta':S([None]*4),'gather':S([None]*4),'sum':R(0)};out=[];products=0
 def add(i,t,e,r,kind=None,sub=None,tokens=None):out.append(step(i,t,e,st,r,kind,sub,metrics=[metric('products','Executed scalar products',products,'products'),metric('valueBytes','Compressed INT8 values',len(sum(packed,[])),'bytes'),metric('metadataBits','Two position bits per retained value',sum(len(p) for p in positions)*2,'bits')],tokens=tokens))
 add('start','Start from a valid 2:4 row','Each consecutive group of four already contains exactly two nonzero weights.','The lesson executes an existing sparse row; it does not prune a dense model.')
 st['packed']=S(sum(packed,[]));st['meta']=S(sum(positions,[]))
 add('encode','Pack retained weights and position metadata','Each two-bit position identifies a retained value within its four-position group.','The compressed value payload is smaller, with extra metadata traffic.','transform',['packed','meta'])
 for j,(p,b) in enumerate(zip(positions,packed)):
  selected=[x[j*4+k] for k in p];st['gather']['values'][j*2:j*2+2]=selected
  add('gather'+str(j),'Select activations for group '+str(j+1),'The position tags select matching dense input elements; the input vector is unchanged.','Nonzero weights now align with the correct operand values.','copy',['x'],[cp('selected'+str(j)+str(k),v,anchor('x',j*4+p[k]),anchor('gather',j*2+k)) for k,v in enumerate(selected)])
  old=st['sum']['value'];st['sum']=R(old+sum(a*v for a,v in zip(b,selected)),'Sparse group accumulated', ' + '.join([str(old)]+[f'{a} × {v}' for a,v in zip(b,selected)]),old);products+=len(b)
  add('compute'+str(j),'Accumulate sparse group '+str(j+1),'Multiply only the two retained weights by their selected activations.','The final sparse sum equals the dense dot product of the already-pruned row.' if j==len(groups)-1 else 'The wide sum remains available for the next group.','execute',['sum'])
 assert st['sum']['value']==dense
 return lesson('fig-2-4-gemm','Position tags select the matching operands','Why are retained values insufficient without their original positions?','Toy eight-element INT8 row, two valid groups of four. Values are already pruned. Each retained value has a two-bit position tag, as in the chapter’s teaching description. This demonstrates arithmetic equivalence, not measured hardware speed or pruning quality.','representation',O,out,['Each four-position group retains exactly two nonzero values.','Metadata positions are relative to their own group, not global vector positions.','Dense operand reads are non-destructive.','Sparse and dense dot products agree for this already-pruned row; end-to-end speedup is not inferred.'],[{'kind':'constantValues','object':'weights'},{'kind':'constantValues','object':'x'}])

def quantization():
 # Negative dot product makes the following ReLU a genuine value transformation.
 x=[.5,-1.];w=[.5,1.5];sx=sw=.5;sy=.25;zx=2;zw=-1;zy=0
 qx=[round(v/sx)+zx for v in x];qw=[round(v/sw)+zw for v in w];acc=0
 O=[seq('realx','Floating-point activations',2,0),seq('realw','Floating-point weights',2,0,1),seq('qx','INT8 activations: scale 0.5, zero point 2',2,1),seq('qw','INT8 weights: scale 0.5, zero point −1',2,1,1),reg('acc','INT32 accumulator',2,typ='INT32',unit='integer units'),reg('out','INT8 output: scale 0.25, zero point 0',2,1,'INT8','code')]
 st={'realx':S(x),'realw':S(w),'qx':S([None,None]),'qw':S([None,None]),'acc':R(0),'out':R(None)};out=[];count=0
 def add(i,t,e,r,kind=None,sub=None):out.append(step(i,t,e,st,r,kind,sub,metrics=[metric('products','Accumulated products',count,'products')]))
 add('start','Begin a two-element dot product','Scales and zero points define how integer codes represent the floating-point operands.','The wide accumulator begins at zero; the output is unwritten.')
 st['qx']=S(qx);st['qw']=S(qw)
 add('quantize','Quantize both operand vectors','Store round(real/scale) + zero point in INT8 for every operand.','Stored codes remain distinct from the real values they represent.','transform',['qx','qw'])
 for j in range(len(x)):
  old=acc;acc+=(qx[j]-zx)*(qw[j]-zw);count+=1;st['acc']=R(acc,'Wide running sum',f'{old} + ({qx[j]} − {zx}) × ({qw[j]} − ({zw}))',old)
  add('mac'+str(j),'Accumulate operand pair '+str(j+1),'Read both codes, subtract their zero points, and accumulate the product into INT32.','The operand codes remain resident while the wide sum changes.','execute',['acc'])
 assert math.isclose(acc*sx*sw,sum(a*b for a,b in zip(x,w)))
 qy=max(-128,min(127,round(acc*sx*sw/sy)+zy));st['out']=R(qy,'Requantized before activation',f'round({acc} × {sx} × {sw} / {sy}) + {zy}')
 add('requantize','Requantize the completed sum','Apply the effective scale and clamp the complete result to INT8 before activation.','The output code represents the negative dot product.','transform',['out'])
 st['out']=R(max(qy,zy),'ReLU after requantization',f'max({qy}, {zy})',qy)
 add('relu','Apply ReLU in output-code space','Real zero maps to the declared output zero point; ReLU clamps codes below it.','The final INT8 output represents real zero.','transform',['out'])
 return lesson('fig-weight-activations-quantization','Codes, wide sums, and the output conversion','Why does an INT8 dot product retain a wider accumulator?','Toy two-element dot product with exactly represented inputs; activation and weight scales are 0.5 and zero points are 2 and −1. Output scale is 0.25 with zero point 0. Signed INT8 clamping precedes ReLU, matching the source pipeline. No throughput claim.','reduction',O,out,['Input codes retain their original values during accumulation.','Products use zero-point-corrected operands and accumulate into INT32.','The complete sum is requantized before ReLU.','INT8 describes stored operand/output codes; INT32 describes the retained sum.'],[{'kind':'constantValues','object':'realx'},{'kind':'constantValues','object':'realw'}])

def build():return [checkpoint(),prefetch(),accumulation(),attention(),selection(),block_sparse(),calibration(),quantization(),sparsity24()]

def main():
 expected=build();path=pathlib.Path(__file__).with_name('vol1-b.json')
 if '--write' in sys.argv:path.write_text(json.dumps(expected,ensure_ascii=False,indent=2)+'\n')
 actual=json.loads(path.read_text());assert actual==expected,'Authored JSON differs from recomputed declared inputs/lifecycles'
 for l in actual:
  for case,steps in [('original',l['steps'])]+[(v['id'],v['steps']) for v in l.get('variants',[])]:
   # Assert every action changes semantic state rather than status text alone.
   for a,b in zip(steps,steps[1:]):
    def semantic(s):
     d=copy.deepcopy(s['snapshot']['objects'])
     for state in d.values():state.pop('status',None)
     return d,s['snapshot']['tokens']
    assert semantic(a)!=semantic(b),(l['key'],b['id'],'No mechanism change')
   final=steps[-1]
   endpoints={o['id']:final['snapshot']['objects'][o['id']].get('value',final['snapshot']['objects'][o['id']].get('values')) for o in l['objects'] if o['kind']=='register' or o['role']=='output'}
   print(l['key'],case,final['id'],json.dumps({'endpoints':endpoints,'metrics':final['snapshot']['metrics']},ensure_ascii=False))
 print('PASS: all snapshots, variants, arithmetic, counts, and declared toy lifecycle endpoints recomputed; default mode performs no writes.')
if __name__=='__main__':main()

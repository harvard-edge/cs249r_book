#!/usr/bin/env python3
"""Derive all lesson snapshots from explicit inputs; default checks without writes."""
import copy,json,pathlib,sys

def obj(id,kind,label,role,config,n):
 return dict(id=id,kind=kind,label=label,role=role,config=config,layout={'wide':{'row':n//2,'column':n%2},'narrow':{'row':n,'column':0}})
def matrix(id,label,r,c,n,role='input'):return obj(id,'matrix',label,role,{'rows':r,'columns':c},n)
def seq(id,label,k,n,role='input'):return obj(id,'sequence',label,role,{'slots':k},n)
def pool(id,label,k,n):return obj(id,'pool',label,'memory',{'slots':k,'columns':min(k,6)},n)
def reg(id,label,n,unit='integer units'):return obj(id,'register',label,'compute',{'type':'Scalar','unit':unit},n)
def state(v,status='',kind='matrix',window=None):
 d={'values':v,'status':status}
 if window is not None:d['window']=window
 return d
def rv(v,status='',expression=None):
 d={'value':v,'status':status}
 if expression:d['expression']=expression
 return d
def ps(values,owner='Model'):
 return {'slots':[None if v is None else {'value':v,'owner':owner,'status':'retained'} for v in values],'status':''}
def metric(id,label,value,unit):return dict(id=id,label=label,value=value,unit=unit)
def step(id,title,exp,states,metrics,result,kind=None,subjects=None,tokens=None,fr=None,to=None):
 t=None if kind is None else dict(kind=kind,subjects=subjects or [],meaning=exp)
 if fr:t['from']=fr
 if to:t['to']=to
 return dict(id=id,title=title,explanation=exp,snapshot=dict(objects=copy.deepcopy(states),tokens=copy.deepcopy(tokens or []),metrics=metrics,activeConnections=[],result=result),transition=t)
def lesson(key,title,question,scope,family,objects,steps,invariants,checks=None):
 d=dict(schema=1,key='vol1:'+key,title=title,question=question,scope=scope,family=family,objects=objects,steps=steps,invariants=invariants)
 if checks:d['checks']=checks
 return d
def copytoken(id,value,source,dest,owner='Model'):
 return dict(id=id,label=str(value),value=value,role='input',owner=owner,at=dest,state='copy',sourceAt=source)
def build():
 lessons=[]
 # Batched matrix multiplication: explicit primitive toy tensors.
 X=[[1,2],[3,4]]; W=[[2,1],[1,3]]; rows=len(X); width=len(W); cols=len(W[0])
 Z=[[sum(X[i][k]*W[k][j] for k in range(width)) for j in range(cols)] for i in range(rows)]
 os=[matrix('x','Input batch X',rows,width,0),matrix('w','Weights in source memory',width,cols,1),matrix('tile','Retained weight tile',width,cols,2,'memory'),seq('row','Current input row',width,3),matrix('z','Output batch Z',rows,cols,4,'output')]
 s={'x':state(sum(X,[])),'w':state(sum(W,[])),'tile':state([None]*(width*cols)),'row':state([None]*width),'z':state([None]*(rows*cols))}; ss=[]
 def bm(fetch,mac):return [metric('fetched','Distinct weight values fetched',fetch,'values'),metric('macs','Completed work',mac,'MACs')]
 ss.append(step('ready','Establish the batch and weights','The toy layer has two input features and two output neurons; biases and activation functions are omitted.',s,bm(0,0),'No output has been computed.'))
 s['tile']=state(sum(W,[])); toks=[copytoken('wcopy'+str(k),v,{'object':'w','index':k},{'object':'tile','index':k}) for k,v in enumerate(sum(W,[]))]
 ss.append(step('load','Copy the weights into the tile','One tile load retains all weights for both examples.',s,bm(width*cols,0),'The source weights remain unchanged.','copy',['w'],toks,{'object':'w'},{'object':'tile'}))
 for i,row in enumerate(X):
  s['row']=state(row); toks=[copytoken('xcopy'+str(k),v,{'object':'x','index':i*width+k},{'object':'row','index':k}) for k,v in enumerate(row)]
  ss.append(step('row'+str(i),'Read input row '+str(i),'The next example reads the same retained weight tile.',s,bm(width*cols,i*width*cols),'The weight tile does not reload.','copy',['x'],toks,{'object':'x','index':i*width},{'object':'row','index':0}))
  s['z']['values'][i*cols:(i+1)*cols]=Z[i]
  ss.append(step('compute'+str(i),'Compute the output row','The row dot products complete together across the two output neurons.',s,bm(width*cols,(i+1)*width*cols),'Output row '+str(i)+' is '+str(Z[i])+'.','execute',['z','tile']))
 lessons.append(lesson('fig-neuron-to-machine','Reuse one weight tile across examples','What changes when a second example uses the same weights?','Toy 2×2 matrix multiplication. One retained tile fits all weights; input/output traffic and measured performance are outside this example.','spatial',os,ss,['Each output is a row–column dot product.','Input rows and source weights remain unchanged.','A weight fetch and a MAC are different events.'],[{'kind':'constantValues','object':'x'},{'kind':'constantValues','object':'w'},{'kind':'nondecreasingMetric','metric':'macs'}]))
 # Source-faithful MNIST memory lifecycle; source primitive dimensions checked locally.
 root=next((p for p in pathlib.Path(__file__).resolve().parents if (p/'books/vol1').exists()),None)
 if root:
  import ast,re
  source=(root/'books/vol1/05_nn_computation/05_nn_computation.qmd').read_text()
  dims=ast.literal_eval(re.search(r'^_MNIST_LAYERS_DIMS = (.+)$',source,re.M).group(1)); batch=int(re.search(r'^_MNIST_BATCH_SIZE = (.+)$',source,re.M).group(1))
 else:dims=[784,128,64,10];batch=32
 # FP32 width follows the representation definition (32 bits / 8 bits per byte).
 bpp=32//8; sizes=[batch*d*bpp for d in dims]; params=[(a*b+b)*bpp for a,b in zip(dims,dims[1:])]; total=sum(params)
 os=[pool('persistent','Weights and Adam buffers',3,0),pool('acts','Saved activation tensors',len(dims),1),pool('grads','Parameter-gradient buffers',len(params),2),seq('work','Backward consumers / update',len(params)+1,3,'compute')]
 os[1]['config']['columns']=2
 s={'persistent':ps(['W: '+str(total)+' B','Adam m: '+str(total)+' B','Adam v: '+str(total)+' B']),'acts':ps(['a'+str(i)+': '+str(v)+' B' for i,v in enumerate(sizes)],'Training step'),'grads':ps([None]*len(params),'Training step'),'work':state(['waiting']*(len(params)+1))};ss=[]
 def mm():
  act=sum(v for i,v in enumerate(sizes) if s['acts']['slots'][i]); grad=sum(v for i,v in enumerate(params) if s['grads']['slots'][i]);return [metric('activations','Saved activations',act,'B'),metric('gradients','Gradient buffers',grad,'B'),metric('total','Live modeled memory',3*total+act+grad,'B')]
 ss.append(step('forward','Begin at forward completion','All four saved tensors are live. Weights and both Adam buffers remain resident throughout the step.',s,mm(),'Backward consumers have not completed.'))
 for l in reversed(range(len(params))):
  s['grads']['slots'][l]=ps(['dW'+str(l)+': '+str(params[l])+' B'],'Training step')['slots'][0]; s['work']['values'][l]='done'
  ss.append(step('backward'+str(l),'Finish backward layer '+str(l+1),'This layer computes its parameter gradients and propagates the activation gradient. Its saved tensors are still retained at this endpoint.',s,mm(),'The completed consumer permits a separate release.','execute',['grads','work']))
  s['acts']['slots'][l+1]=None
  ss.append(step('release'+str(l),'Release saved output a'+str(l+1),'The layer output has no remaining backward consumer in this sequential network.',s,mm(),'The released activation storage is now free.','release',['acts']))
 s['acts']['slots'][0]=None
 ss.append(step('release-input','Release the input batch','Backward layer 1 has finished its final read of the input batch.',s,mm(),'Only parameter state remains in this accounting.','release',['acts']))
 s['work']['values'][-1]='done';s['persistent']['slots'][0]['value']='W updated: '+str(total)+' B'
 ss.append(step('update','Apply the Adam update','The optimizer changes parameter values while the gradient buffers remain allocated, matching the source timeline.',s,mm(),'Weights, two Adam buffers, and allocated gradients persist.','execute',['persistent','work']))
 lessons.append(lesson('fig-training-step-memory','Release activations after their last consumers','Why does activation storage fall while gradient storage grows?','Source network 784–128–64–10, batch 32, FP32, decimal bytes. Starts at forward completion. Pool cells represent whole tensors of unequal sizes; their widths do not encode bytes. Workspace and allocator overhead are omitted.','ownership',os,ss,['Weights and both Adam buffers persist.','A saved activation is released only after its backward consumer completes.','The final source-figure accounting retains allocated gradient buffers.']))
 # Cross-correlation, first output row only: parallel products then reduction.
 inp=list(range(1,10));kernel=[1,0,0,1]; os=[matrix('input','Input',3,3,0),matrix('kernel','Shared kernel',2,2,1),matrix('products','Weighted contributions',2,2,2,'compute'),reg('acc','Retained patch sum',3),matrix('output','Output positions',2,2,4,'output')];s={'input':state(inp),'kernel':state(kernel),'products':state([None]*4),'acc':rv(None),'output':state([None]*4)};ss=[]
 def cm():return [metric('written','Written output positions',sum(v is not None for v in s['output']['values']),'positions')]
 ss.append(step('ready','Establish input and shared kernel','The output and contribution cells begin unwritten; a null cell is not numerical zero.',s,cm(),'Stride 1 and valid boundaries produce a 2×2 output.'))
 for c in range(2):
  s['input']['window']={'row':0,'column':c,'rows':2,'columns':2}
  ss.append(step('window'+str(c),'Select receptive field '+str(c),'Move the footprint without moving or erasing any input value. The register retains its previous result until the new reduction finishes.',s,cm(),'The selected patch supplies the next four paired multiplications.','window',['input']))
  indices=[r*3+c+k for r in range(2) for k in range(2)];patch=[inp[i] for i in indices];products=[a*b for a,b in zip(patch,kernel)];terms=[str(a)+'×'+str(b)+' = '+str(v) for a,b,v in zip(patch,kernel,products)];s['products']=state(products)
  tokens=[dict(id='term'+str(c)+'-'+str(i),label=term.replace(' = ',' → '),value=products[i],role='compute',owner='',at={'object':'products','index':i},state='inflight',sourceAt={'object':'input','index':indices[i]}) for i,term in enumerate(terms)]
  ss.append(step('products'+str(c),'Form four weighted contributions','Each input in the footprint multiplies the kernel value at the same local position. All four multiplications form in parallel; the old register sum remains retained.',s,cm(),'The contribution cells expose every product, including multiplication by zero.','transform',['products','input','kernel'],tokens))
  v=sum(products);s['acc']=rv(v,expression=' + '.join(map(str,products)))
  ss.append(step('compute'+str(c),'Reduce the four products into the sum','Add the four weighted contributions. The completed sum replaces the previous register value only when this reduction finishes.',s,cm(),'The retained sum is '+str(v)+'; its output cell is still unwritten.','execute',['products','acc'],fr={'object':'products'},to={'object':'acc'}))
  s['output']['values'][c]=v
  ss.append(step('write'+str(c),'Write output position '+str(c),'Commit the completed sum into its corresponding output cell.',s,cm(),'Output (0, '+str(c)+') now stores '+str(v)+'. The input and shared kernel remain unchanged.','write',[],fr={'object':'acc'},to={'object':'output','index':c}))
 lessons.append(lesson('fig-cnn','Slide a shared spatial kernel','How does one receptive field determine one output cell?','Toy 3×3 input and 2×2 kernel, stride 1, valid padding. Shows only the first output row. Uses CNN cross-correlation convention: the kernel is not flipped.','spatial',os,ss,['Source input and shared kernel remain unchanged.','Every contribution pairs equal local positions in the input patch and kernel.','Only a completed reduction writes an output cell.'],[{'kind':'constantValues','object':'kernel'},{'kind':'metricEqualsNonNull','metric':'written','object':'output'}]))
 lessons[-1]['narrowColumns']=2
 for o in lessons[-1]['objects']:
  o['layout']['narrow']=dict(o['layout']['wide'])
  if o['id']=='output':o['layout']['narrow']={'row':2,'column':0,'columnSpan':2}
 # Factorized channel filtering: independent spatial reductions then mixing.
 channels=[[1,2,3,4],[2,1,0,1],[1,0,2,1]]; filters=[[1,0,0,1],[0,1,1,0],[1,1,0,0]]; mix=[[1,2],[2,0],[0,1]]
 intermediates=[sum(a*b for a,b in zip(ch,f)) for ch,f in zip(channels,filters)];outs=[sum(intermediates[c]*mix[c][o] for c in range(len(channels))) for o in range(len(mix[0]))]
 os=[matrix('patch','One patch per input channel',3,4,0),matrix('depth','Separate spatial filters',3,4,1),seq('mid','Depthwise intermediates',3,2,'output'),matrix('mix','Pointwise mixing weights',3,2,3),seq('out','Output channels at this position',2,4,'output')]
 s={'patch':state(sum(channels,[])),'depth':state(sum(filters,[])),'mid':state([None]*len(channels)),'mix':state(sum(mix,[])),'out':state([None]*len(outs))};ss=[]
 def dm():return [metric('spatial','Completed spatial MACs',sum(v is not None for v in s['mid']['values'])*len(channels[0]),'MACs'),metric('mixing','Completed mixing MACs',sum(v is not None for v in s['out']['values'])*len(channels),'MACs')]
 ss.append(step('ready','Establish independent channel patches','Each row holds one flattened 2×2 spatial patch and its own depthwise filter.',s,dm(),'No channel has yet been mixed with another.'))
 for c,v in enumerate(intermediates):
  s['mid']['values'][c]=v
  ss.append(step('depth'+str(c),'Filter input channel '+str(c+1),'This spatial dot product reads only its own channel.',s,dm(),'Channel '+str(c+1)+' produces intermediate '+str(v)+'.','execute',['mid']))
 for o,v in enumerate(outs):
  s['out']['values'][o]=v
  ss.append(step('mix'+str(o),'Mix channels for output '+str(o+1),'One column of the 1×1 mixer combines all depthwise intermediates.',s,dm(),'Output '+str(o+1)+' is '+str(v)+'.','execute',['out']))
 lessons.append(lesson('fig-depthwise-pointwise-channels','Separate spatial filtering from channel mixing','When do values from different channels first combine?','Toy factorized layer at one spatial position: three 2×2 patches, depth multiplier one, two output channels, no biases. Flattened rows preserve spatial order. This factorization is not asserted equal to an arbitrary standard convolution.','reduction',os,ss,['Depthwise filtering does not sum across channels.','Each pointwise output sums across all intermediate channels.','Zero weights still count as dense MACs.']))
 # Materialized im2col with exact source-art numbers.
 maps=[list(range(1,10)),list(range(10,19))]; weights=list(range(1,9)); patches=[]
 for r in range(2):
  for c in range(2):patches.append([m[(r+dr)*3+c+dc] for m in maps for dr in range(2) for dc in range(2)])
 outputs=[sum(a*b for a,b in zip(row,weights)) for row in patches];os=[matrix('a','Input channel 1',3,3,0),matrix('b','Input channel 2',3,3,1),matrix('lower','Materialized patch rows',4,8,2,'memory'),matrix('weights','Stacked filter weights',8,1,3),matrix('output','Spatial output',2,2,4,'output')]
 s={'a':state(maps[0]),'b':state(maps[1]),'lower':state([None]*(len(patches)*len(weights))),'weights':state(weights),'output':state([None]*len(outputs))};ss=[]
 def im():return [metric('materialized','Materialized patch values',sum(v is not None for v in s['lower']['values']),'values'),metric('source','Source values retained',sum(map(len,maps)),'values')]
 ss.append(step('ready','Establish both input channels','Each output patch includes a 2×2 region from each input channel.',s,im(),'Source elements remain resident throughout lowering.'))
 for i,row in enumerate(patches):
  r,c=divmod(i,2)
  for id in ['a','b']:s[id]['window']={'row':r,'column':c,'rows':2,'columns':2}
  s['lower']['values'][i*len(weights):(i+1)*len(weights)]=row
  tokens=[]
  for j,v in enumerate(row):
   ch,local=divmod(j,4);dr,dc=divmod(local,2);tokens.append(copytoken('patch'+str(i)+'v'+str(j),v,{'object':['a','b'][ch],'index':(r+dr)*3+c+dc},{'object':'lower','index':i*len(weights)+j}))
  ss.append(step('copy'+str(i),'Copy receptive field '+str(i)+' into its row','Both channels contribute in fixed channel-major, then row-major order; overlapping source values are copied again.',s,im(),'The source maps are unchanged; this row owns distinct copied storage.','copy',['a','b'],tokens,{'object':'a'},{'object':'lower'}))
 s['output']=state(outputs)
 ss.append(step('gemm','Compute the lowered matrix product','Each lowered row forms one dot product with the stacked filter column.',s,im(),'The row order maps to spatial output values '+str(outputs)+'.','execute',['output']))
 lessons.append(lesson('fig-im2col-diagram','Copy overlapping patches into GEMM rows','Why does explicit lowering store repeated source values?','Exact artwork inputs: two 3×3 channels and stacked filter values 1–8; stride 1, valid 2×2 patches. Fully materialized lowering only. GEMM writes the four results in spatial row-major order.','representation',os,ss,['Copying a patch never deletes its source elements.','All lowered rows use identical channel and spatial ordering.','Implicit GEMM need not allocate this materialized buffer.'],[{'kind':'constantValues','object':'a'},{'kind':'constantValues','object':'b'},{'kind':'metricEqualsNonNull','metric':'materialized','object':'lower'}]))
 # Prefill and one decode: attention math abstracted, ownership explicit.
 prompt=['The','cat','sat','on'];owner='Request A';cache_size=len(prompt)+1
 os=[seq('prompt','Known prompt positions',len(prompt),0),pool('cache','Persistent per-layer K/V cache',cache_size,1),obj('current','chamber','Current-pass state','compute',{'slots':5},2),seq('scratch','Read copies of prior K/V',len(prompt),3,'memory'),seq('generated','Generated tokens',2,4,'output')]
 os[1]['config']['columns']=3
 s={'prompt':state(prompt),'cache':ps([None]*cache_size,owner),'current':state([None]*5),'scratch':state([None]*len(prompt)),'generated':state([None]*2)};ss=[]
 def km():return [metric('occupied','Cached positions',sum(v is not None for v in s['cache']['slots']),'positions'),metric('emitted','Emitted tokens',sum(v is not None for v in s['generated']['values']),'tokens')]
 ss.append(step('ready','Establish the complete prompt','All prompt positions are available together; causal attention still restricts each position to itself and earlier positions.',s,km(),'The cache is empty and generation has not started.'))
 s['current']=state(['K'+str(i)+'/V'+str(i) for i in range(len(prompt))]+['logits'])
 ss.append(step('prefill','Compute the prompt-wide pass','Prefill evaluates all known prompt positions with causal attention; the final prompt position supplies next-token logits.',s,km(),'Prompt K/V exist but are not yet stored persistently.','execute',['current']))
 s['cache']=ps(['K'+str(i)+'/V'+str(i) for i in range(len(prompt))]+[None],owner)
 ss.append(step('store-prompt','Store all prompt K/V entries','The cache retains one entry for every consumed prompt position.',s,km(),'These entries will survive the next decode pass.','write',['cache'],fr={'object':'current'},to={'object':'cache'}))
 s['generated']['values'][0]='the'
 ss.append(step('emit-first','Emit the first generated token','Select the first generated token from the prefill logits.',s,km(),'The token “the” has been emitted but has no cache entry yet.','reveal',['generated']))
 s['current']=state(['the','q4','K4','V4',None]); toks=[copytoken('feedback','the',{'object':'generated','index':0},{'object':'current','index':0},owner)]
 ss.append(step('consume','Compute current-token projections','Decode consumes the emitted token and computes only its current query, key, and value.',s,km(),'Prior prompt K/V remain cached.','execute',['current'],toks))
 s['scratch']['values']=['K'+str(i)+'/V'+str(i) for i in range(len(prompt))]; toks=[copytoken('read'+str(i),'K'+str(i)+'/V'+str(i),{'object':'cache','index':i},{'object':'scratch','index':i},owner) for i,w in enumerate(prompt)]
 ss.append(step('read','Read retained prior K/V','Read the prior entries without evicting them; the current key/value are already available in the current-pass state.',s,km(),'Attention has the past state and the current position.','copy',['cache'],toks,{'object':'cache'},{'object':'scratch','index':0}))
 s['current']['values'][4]='logits'
 ss.append(step('attention','Compute attention and the remaining decode pass','The current query attends to retained prior positions and its own current key/value; the rest of the pass produces next-token logits.',s,km(),'No prior-token projections were recomputed.','execute',['current']))
 s['cache']['slots'][-1]=ps(['K4/V4'],owner)['slots'][0]
 ss.append(step('append','Append the consumed token’s K/V','Retain the current position for future decode steps.',s,km(),'Cache occupancy grows only after “the” has been consumed.','write',['cache'],fr={'object':'current'},to={'object':'cache','index':cache_size-1}))
 s['generated']['values'][1]='mat'
 ss.append(step('emit-next','Emit the next generated token','Select the next token from the completed decode logits.',s,km(),'“mat” will become the following input; its K/V are not cached yet.','reveal',['generated']))
 lessons.append(lesson('fig-prefill-decode','Retain prompt state and extend it during decode','Why is an emitted token cached only on its next input pass?','One request, prompt positions 0–3, one schematic layer cache, one decode pass at position 4. K0/V0 identifies the first prompt position; q4/K4/V4 belong to consumed token “the”. K/V vectors and attention arithmetic are abstracted into named states; no physical byte or timing model.','cache',os,ss,['A K/V entry belongs to a consumed token position.','Reading cached K/V preserves all resident entries.','The current query attends to prior positions and itself.'],[{'kind':'constantValues','object':'prompt'},{'kind':'metricEqualsOccupied','metric':'occupied','object':'cache'}]))
 # Allocator geometry with exact source addresses.
 capacity=12; owners=['A']*4+['B']*4+['C']*4;request=6
 os=[pool('pool','Reserved virtual-address pool',capacity,0),seq('request','Contiguous allocation request',request,1),obj('decision','lanes','Allocation attempt','control',{'lanes':['Allocator'],'start':0,'end':1,'axisLabel':'Logical attempt'},2)]
 os[0]['config']['columns']=capacity
 s={'pool':ps(owners,'Allocator'),'request':state([None]*request),'decision':{'segments':[],'status':''}};ss=[]
 def am():
  vals=s['pool']['slots'];runs=[];run=0
  for v in vals:
   if v is None:run+=1
   else:runs.append(run);run=0
  runs.append(run);free=sum(v is None for v in vals)
  return [metric('reserved','Reserved pool',len(vals),'units'),metric('live','Live tensor storage',len(vals)-free,'units'),metric('free','Reusable storage',free,'units'),metric('largest','Largest contiguous hole',max(runs),'units')]
 for i,o in enumerate(owners):s['pool']['slots'][i]['owner']=o
 ss.append(step('ready','Establish three live tensor blocks','Equal cells are equal units of increasing virtual address. The allocator reserves the entire fixed pool.',s,am(),'A, B, and C each occupy a contiguous block.'))
 for i,o in enumerate(owners):s['pool']['slots'][i]['owner']=o
 for i in range(4,8):s['pool']['slots'][i]=None
 ss.append(step('release-b','Release tensor B','B has no remaining consumer. Its block becomes reusable inside the reserved pool.',s,am(),'Device reservation does not shrink.','release',['pool']))
 for i in range(4,6):s['pool']['slots'][i]={'value':'D','owner':'D','status':'allocated'}
 ss.append(step('allocate-d','Allocate D from part of the cached block','The allocator splits B’s old block and assigns its first two units to D.',s,am(),'The remaining part of B’s former block stays reusable.','write',['pool']))
 for i in range(4):s['pool']['slots'][i]=None
 ss.append(step('release-a','Release tensor A','A has no remaining consumer; D and C stay live at their existing addresses.',s,am(),'Two separated holes now remain.','release',['pool']))
 s['request']=state(['E']*request)
 ss.append(step('request','Submit a contiguous request for E','E requires one uninterrupted run of six units.',s,am(),'Aggregate free space cannot bridge live D.','reveal',['request']))
 s['decision']={'segments':[{'id':'attempt','lane':0,'start':0,'end':1,'label':'E cannot fit','state':'blocked','role':'control'}],'cursor':1,'status':'No sufficiently large contiguous hole'}
 ss.append(step('blocked','Reject the allocation attempt','The allocator finds no contiguous hole large enough under the fixed-pool policy.',s,am(),'The request fails while all existing addresses and live tensors remain intact.','execute',['attempt']))
 lessons.append(lesson('fig-allocator-live-cached-blocks','Expose fragmented reusable space','Why can a contiguous request fail with enough aggregate free space?','Exact twelve-unit fixed virtual-address pool from the artwork. A, B, C, and D name the tensor owners. No pool growth, compaction, or remapping. A null cell is reusable allocator storage; it is still part of the reserved pool.','ownership',os,ss,['Reservation remains constant.','Allocation requires a contiguous run of reusable addresses.','Live tensors do not move to satisfy this request.']))
 # Dispatch: bounded toy schedule, equal arithmetic without invented real specs.
 dispatch=2;compute=1;nops=2;end=nops*(dispatch+compute)
 os=[obj('eager','lanes','Eager operation chain','compute',{'lanes':['Host dispatch','Device compute'],'start':0,'end':end,'axisLabel':'Toy schedule units'},0),obj('fused','lanes','Fused operation chain','compute',{'lanes':['Host submission','Device compute'],'start':0,'end':end,'axisLabel':'Toy schedule units'},1),seq('eout','Eager arithmetic outputs',nops,2,'output'),seq('fout','Fused arithmetic outputs',nops,3,'output')]
 def seg(id,lane,start,end,label):return dict(id=id,lane=lane,start=start,end=end,label=label,state='waiting',role='control' if lane==0 else 'compute')
 e=[]
 for i in range(nops):
  t=i*(dispatch+compute);e.extend([seg('d'+str(i),0,t,t+dispatch,'dispatch '+str(i+1)),seg('k'+str(i),1,t+dispatch,t+dispatch+compute,'operation '+str(i+1))])
 f=[seg('fd',0,0,dispatch,'one submission'),seg('fk',1,dispatch,dispatch+nops*compute,'fused arithmetic')]
 s={'eager':{'segments':e,'cursor':0,'status':''},'fused':{'segments':f,'cursor':0,'status':''},'eout':state([None]*nops),'fout':state([None]*nops)};ss=[];inputs=[3,2];outputs=[inputs[0]+inputs[1],(inputs[0]+inputs[1])*inputs[1]]
 def pm():return [metric('ed','Eager submissions complete',sum(x['state']=='done' and x['lane']==0 for x in s['eager']['segments']),'submissions'),metric('fd','Fused submissions complete',sum(x['state']=='done' and x['lane']==0 for x in s['fused']['segments']),'submissions'),metric('results','Arithmetic results available',sum(v is not None for v in s['eout']['values'])+sum(v is not None for v in s['fout']['values']),'values')]
 ss.append(step('ready','Establish the same dependent arithmetic','Both cases compute (3 + 2), then multiply that result by 2. The fused implementation preserves that order within one kernel.',s,pm(),'The bars are planned tasks, initially waiting.'))
 for i in range(nops):
  s['eager']['segments'][2*i]['state']='done';s['eager']['cursor']=(i*(dispatch+compute))+dispatch
  ss.append(step('dispatch'+str(i),'Complete eager dispatch '+str(i+1),'The host finishes preparing and submitting this device operation; its device result does not exist yet.',s,pm(),'The corresponding device operation can now execute.','execute',['d'+str(i)]))
  s['eager']['segments'][2*i+1]['state']='done';s['eager']['cursor']=(i+1)*(dispatch+compute);s['eout']['values'][i]=outputs[i]
  ss.append(step('kernel'+str(i),'Complete eager operation '+str(i+1),'The device completes the submitted operation. A later dependent eager operation still requires its own host dispatch.',s,pm(),'The retained result is '+str(outputs[i])+'.','execute',['k'+str(i),'eout']))
 s['fused']['segments'][0]['state']='done';s['fused']['cursor']=dispatch
 ss.append(step('fused-dispatch','Complete the single fused submission','One compiled submission describes both arithmetic operations.',s,pm(),'The device has the full dependent operation chain.','execute',['fd']))
 s['fused']['segments'][1]['state']='done';s['fused']['cursor']=dispatch+nops*compute;s['fout']=state(outputs)
 ss.append(step('fused-kernel','Complete the fused kernel','The kernel performs both dependent operations without an intermediate host submission.',s,pm(),'Both cases retain the same final arithmetic result.','execute',['fk','fout']))
 lessons.append(lesson('fig-python-tax','Expose host gaps between device operations','Which dependency disappears when operations share one submission?','Toy two-operation schedule: dispatch takes 2 abstract units; each arithmetic operation takes 1. Fusion is assumed legal and retains equal arithmetic cost. The source figure’s five-operation physical timing model is not reproduced here. Playback duration is unrelated to schedule units.','schedule',os,ss,['Fusing preserves arithmetic order and the final result.','Device work depends on its preceding submission.','The toy schedule is not a measured speedup or a universal performance model.']))
 # Logical coordinates versus two immutable physical orders.
 logical=[[1,2,3],[4,5,6]];R=len(logical);C=len(logical[0]);rowmajor=sum(logical,[]);columnmajor=[logical[r][c] for c in range(C) for r in range(R)];width_bytes=32//8
 os=[matrix('logical','Logical tensor',R,C,0),seq('row','Row-major physical storage',R*C,1),seq('col','Column-major physical storage',R*C,2),seq('rread','Row-major read trail',C,3,'output'),seq('cread','Column-major read trail',C,4,'output')]
 s={'logical':state(rowmajor),'row':state(rowmajor),'col':state(columnmajor),'rread':state([None]*C),'cread':state([None]*C)};ss=[]
 def lm(r,c):
  ro=r*C+c;co=r+c*R
  return [metric('ro','Row-major element offset',ro,'elements'),metric('co','Column-major element offset',co,'elements'),metric('rb','Row-major byte offset',ro*width_bytes,'B'),metric('cb','Column-major byte offset',co*width_bytes,'B')]
 for o in os:
  if o['id'] in ['row','col']:o['config']['slotLabels']=['addr '+str(i) for i in range(R*C)]
 s['logical']['window']={'row':0,'column':0,'rows':1,'columns':1}
 ss.append(step('ready','Establish both physical orders','The selected logical coordinate is (0, 0). Element strides are [3, 1] for row-major and [1, 2] for column-major.',s,lm(0,0),'Byte offsets multiply element offsets by the declared four-byte element width.'))
 coords=[(0,0),(0,1),(0,2)]
 for i,(r,c) in enumerate(coords):
  if i:
   s['logical']['window']={'row':r,'column':c,'rows':1,'columns':1}
   ss.append(step('select'+str(i),'Select logical coordinate '+str((r,c)),'Move the logical selection while both physical storage orders remain intact.',s,lm(r,c),'The offset changes according to each storage layout.','window',['logical']))
  ro=r*C+c;co=r+c*R;v=logical[r][c];s['rread']['values'][i]=rowmajor[ro];s['cread']['values'][i]=columnmajor[co];s['rread']['status']='Read address '+str(ro);s['cread']['status']='Read address '+str(co);tokens=[copytoken('readrow',v,{'object':'row','index':ro},{'object':'rread','index':i}),copytoken('readcol',v,{'object':'col','index':co},{'object':'cread','index':i})]
  ss.append(step('read'+str(i),'Read the selected value from both layouts','Each layout reaches the same logical value through its own element offset. Reads preserve the physical arrays.',s,lm(r,c),'Both reads return '+str(v)+'.','copy',['row','col'],tokens,{'object':'row','index':ro},{'object':'rread','index':i}))
 lessons.append(lesson('fig-tensor-memory-layout','Map logical coordinates into physical storage','Which physical layout makes this row traversal adjacent?','Exact 2×3 artwork tensor in row-major and column-major order, traversing the first logical row with declared 32-bit elements. Offsets are relative to storage origin. Selection and reading do not transpose or copy the tensor’s physical storage.','representation',os,ss,['Both layouts represent the same logical tensor.','Element offsets and byte offsets have distinct units.','A read never removes a source element.'],[{'kind':'constantValues','object':'row'},{'kind':'constantValues','object':'col'}]))
 return lessons

def main():
 expected=build(); target=pathlib.Path(__file__).with_name('vol1-a.json')
 if '--emit' in sys.argv:
  print(json.dumps(expected,ensure_ascii=False,indent=2));return
 actual=json.loads(target.read_text());assert actual==expected,'Authored JSON differs from reproducible derivation'
 for l in expected:
  for st in l['steps']:
   print(l['key'],st['id'],', '.join(m['id']+'='+str(m['value'])+' '+m.get('unit','') for m in st['snapshot']['metrics']))
 print('Verified all snapshots and derived arithmetic, occupancy, byte totals, offsets, dot products, and schedule endpoints.')
if __name__=='__main__':main()

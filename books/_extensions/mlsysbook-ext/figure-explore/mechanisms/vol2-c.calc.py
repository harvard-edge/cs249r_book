#!/usr/bin/env python3
"""Declared illustrative inputs, deterministic mechanism authoring, and drift checks.
Default mode only reads vol2-c.json. --write explicitly exports recomputed data.
No hardware constants or performance claims are introduced.
"""
import json, copy, argparse
from pathlib import Path
from fractions import Fraction


def obj(i,kind,label,role,config,n):
 return dict(id=i,kind=kind,label=label,role=role,config=config,layout={'wide':{'row':n//2,'column':n%2,'columnSpan':1},'narrow':{'row':n,'column':0,'columnSpan':1}})
def seq(i,label,n,k,role='input'): return obj(i,'sequence',label,role,{'slots':k},n)
def pool(i,label,n,k): return obj(i,'pool',label,'memory',{'slots':k,'columns':min(k,4),'slotLabels':[str(j) for j in range(k)]},n)
def reg(i,label,n,typ='scalar',unit=''): return obj(i,'register',label,'output',dict(type=typ,unit=unit),n)
def chamber(i,label,n,k): return obj(i,'chamber',label,'compute',{'slots':k},n)
def s(values,status=''): return dict(values=values,status=status)
def r(value,status='',expression=None):
 d=dict(value=value,status=status)
 if expression is not None:d['expression']=expression
 return d
def p(slots,status=''): return dict(slots=slots,status=status)
def slot(v,owner,st='retained'):return dict(value=v,owner=owner,status=st)
def metric(i,label,value,unit):return dict(id=i,label=label,value=value,unit=unit)
def trans(kind,subjects,meaning,fr=None,to=None):
 d=dict(kind=kind,subjects=subjects,meaning=meaning)
 if fr:d['from']=fr
 if to:d['to']=to
 return d
def at(i,j=None):return {'object':i,**({'index':j} if j is not None else {})}
def token(i,v,loc,owner='Request A',state='resident',source=None):
 d=dict(id=i,label=str(v),value=v,role='input',owner=owner,at=loc,state=state)
 if source:d['sourceAt']=source
 return d
def lesson(key,title,question,scope,family,objects,invariants):return dict(schema=1,key='vol2:'+key,title=title,question=question,scope=scope,family=family,objects=objects,steps=[],invariants=invariants)
def step(L,i,title,why,state,metrics,result,t=None,tokens=None):
 L['steps'].append(dict(id=i,title=title,explanation=why,snapshot=dict(objects=copy.deepcopy(state),tokens=copy.deepcopy(tokens or []),metrics=copy.deepcopy(metrics),activeConnections=[],result=result),transition=t))
def count(slots):return sum(x is not None for x in slots)
def build():
 lessons=[]
 # One scoped KV pair represents all K/V coordinates for one position.
 L=lesson('fig-inference-lifecycle','Consume, retain, then generate','When does an emitted token acquire a KV-cache entry?','One request, one abstract layer/head. Prompt IDs 11 and 12; sampled outputs 21 and 22. Each cache cell is a K/V pair for a consumed position. No byte or latency model. Host routing/tokenization are already complete.','cache',[seq('prompt','CPU: tokenized prompt',0,2),pool('cache','GPU: request-owned KV positions',1,3),chamber('work','GPU: current computation',2,4),seq('output','CPU: streamed output IDs',3,2,'output'),seq('weights','GPU: shared model weights',4,1,'reference')],['An output acquires cached K/V only when it is consumed as a model input.','Reading retained K/V leaves every source position unchanged.','Request completion frees request state; shared weights remain resident.'])
 st={'prompt':s([11,12]),'cache':p([None]*3),'work':s([None]*4),'output':s([None]*2),'weights':s(['W shared'])}
 def km():return [metric('cached','Retained KV positions',count(st['cache']['slots']),'positions'),metric('emitted','Streamed outputs',sum(v is not None for v in st['output']['values']),'tokens')]
 step(L,'input','Tokenized prompt is ready','The host has admitted the request and produced IDs; the model has not consumed them.',st,km(),'No request KV state exists yet.')
 st['work']=s(['K0/V0','K1/V1','logits → 21',None],'Prefill completed')
 st['cache']=p([slot('11: K0/V0','Request A'),slot('12: K1/V1','Request A'),None])
 step(L,'prefill','Execute prompt prefill','One prompt pass computes K/V at both prompt positions and logits for the first output.',st,km(),'Prompt K/V is retained while logits for output 21 are available.',trans('execute',['work'],'Compute prompt-position state.'))
 st['output']['values'][0]=21
 step(L,'first','Emit the first output','Sampling uses prefill logits; output 21 has not yet been consumed as an input.',st,km(),'Output 21 is streamed while its cache position is still free.',trans('write',['output'],'Emit output 21.',at('work'),at('output',0)))
 st['work']=s(['q2','21: K2/V2',None,None],'Consumed output 21 as input')
 step(L,'consume','Consume output 21 for decode','The next forward computation forms query, key, and value at the new input position.',st,km(),'Position 2 exists in current computation, but has not been appended.',trans('execute',['work'],'Compute current-token q/K/V.'))
 reads=[token('read0','K0/V0',at('work',2),state='copy',source=at('cache',0)),token('read1','K1/V1',at('work',3),state='copy',source=at('cache',1))]
 step(L,'read','Read prior retained K/V','The decoder reads prior retained K/V without evicting it; the copies are temporary operands, not new cache allocations.',st,km(),'Both prior cache entries remain resident.',trans('copy',['cache'],'Read without eviction.',at('cache'),at('work',2)),reads)
 st['work']['values'][2]='logits → 22'
 step(L,'attention','Compute attention and next-token logits','The current query combines retained prior K/V with the current-position K/V. The resulting logits support the next sample.',st,km(),'Decode logits exist; output 22 has not yet been emitted.',trans('execute',['work'],'Compute attention from read operands and the current position.'))
 st['cache']['slots'][2]=slot('21: K2/V2','Request A')
 step(L,'append','Append the consumed token state','The cache now retains the K/V produced from input 21 for later iterations.',st,km(),'The cache includes prompt positions and consumed output 21.',trans('write',['cache'],'Append current position.',at('work',1),at('cache',2)))
 st['output']['values'][1]=22;st['work']['status']='Request reaches its declared stop'
 step(L,'second','Emit output 22 and stop generation','The decoder emits the next sampled ID; this example ends before output 22 is ever consumed.',st,km(),'Output 22 is visible, but no K/V entry for 22 is created.',trans('write',['output'],'Commit next sampled output.',at('work'),at('output',1)))
 st['cache']=p([None]*3,'Request completed');st['work']=s([None]*4,'No consumers remain')
 step(L,'release','Release completed request state','Completion removes the request-owned cache and temporary computation; the shared weights remain.',st,km(),'All request cache slots are free; W remains resident.',trans('release',['cache','work'],'Release after the final consumer.'))
 L['checks']=[{'kind':'metricEqualsOccupied','metric':'cached','object':'cache'},{'kind':'constantValues','object':'weights'},{'kind':'nondecreasingMetric','metric':'emitted'}];lessons.append(L)
 # Paging: separate reference reservation from the actual C/D pool avoids false identity changes.
 L=lesson('fig-paged-attention','Map logical pages onto scattered blocks','How can four free blocks admit E when no four-block gap exists?','The caption’s eight-block pool. Reference panel A reserves four blocks each for A and B. Main pool starts after other requests finish: C owns 2–3 and D owns 6–7. E has four logical pages; the last page is partly used. One cell means a fixed-size page, not one token.','ownership',[pool('reservation','Panel A: contiguous reservation',0,8),pool('physical','Panels B/C: physical blocks',1,8),seq('logical','E logical pages',2,4),seq('table','E page → physical block',3,4,'control'),chamber('reader','Attention page read',4,1),reg('admit','Contiguous admission',5,'decision')],['Paging changes addressing and allocation, not the size or contents of KV values.','C and D retain their own blocks throughout.','A free block differs from reserved but unused capacity.'])
 reserved=[slot('A used','A') for _ in range(2)]+[slot('unused','A','allocated') for _ in range(2)]+[slot('B used','B') for _ in range(3)]+[slot('unused','B','allocated')]
 physical=[None,None,slot('C','C'),slot('C','C'),None,None,slot('D','D'),slot('D','D')]
 st={'reservation':p(reserved),'physical':p(physical),'logical':s(['E0','E1','E2','E3 partial']),'table':s([None]*4),'reader':s([None]),'admit':r(None)}
 def gaps(a):
  runs=[];run=0
  for x in a+[True]:
   if x is None:run+=1
   else:runs.append(run);run=0
  return max(runs)
 def pm():return [metric('free','Free physical blocks',len(physical)-count(physical),'blocks'),metric('largest','Largest contiguous gap',gaps(physical),'blocks'),metric('waste','Reserved unused reference blocks',sum(x['value']=='unused' for x in reserved),'blocks')]
 step(L,'fragmented','Inspect internal and external fragmentation','The reference reserves capacity never used by A/B. The current pool has two separate free runs.',st,pm(),'E cannot use the sum of gaps under contiguous allocation.')
 st['admit']=r('DENIED','Need one contiguous run of four blocks')
 step(L,'deny','Test contiguous admission','The largest free run is shorter than E’s declared four-page requirement.',st,pm(),'The admission decision changes; no C/D allocation moves.',trans('execute',['admit'],'Evaluate contiguous-fit predicate.'))
 mapping=[i for i,x in enumerate(physical) if x is None]
 st['table']=s(mapping,'Logical pages map to free physical blocks')
 step(L,'map','Create E’s block table','Assign each logical page a distinct free physical block.',st,pm(),'E’s address order is logical rather than physical adjacency.',trans('write',['table'],'Publish logical-to-physical mapping.'))
 for j,i in enumerate(mapping):physical[i]=slot('E'+str(j)+(' partial' if j==len(mapping)-1 else ''),'E')
 st['physical']=p(physical)
 step(L,'allocate','Allocate E’s four mapped blocks','The same four free blocks now belong to E, including the partly filled tail page.',st,pm(),'E fits without moving C or D and without compressing any KV.',trans('write',['physical'],'Allocate using the table.'))
 j=2;i=mapping[j];st['reader']=s(['E'+str(j)],'Logical page 2 read via physical block '+str(i))
 read=[token('e-read','E2',at('reader',0),owner='E',state='copy',source=at('physical',i))]
 step(L,'read','Read logical page 2 through the table','The lookup selects its mapped block; a read preserves physical residency.',st,pm(),'Physical block '+str(i)+' still holds E2 after the read.',trans('copy',['physical'],'Copy one mapped page for attention.',at('physical',i),at('reader',0)),read)
 for i in mapping:physical[i]=None
 st['physical']=p(physical,'E completed');st['table']=s([None]*4,'E mapping released');st['reader']=s([None],'No remaining consumers')
 step(L,'release','Release E after completion','Only blocks owned by E and its now-unused mapping/read state are released.',st,pm(),'The original scattered free capacity returns; C and D remain.',trans('release',['physical','table','reader'],'Free E-owned state after its last consumer.'))
 lessons.append(L)
 # Exact rejection arithmetic for a declared trace, not a general speed claim.
 draft=['T1','T2','T3','T4'];pvals=[.8,.5,.6,.1];qvals=[.4,.5,.3,.5];uniforms=[.3,.2,.7,.8]
 accept=[min(1,pv/qv) for pv,qv in zip(pvals,qvals)];accepted=next((i for i,(u,a) in enumerate(zip(uniforms,accept)) if u>a),len(draft))
 L=lesson('fig-speculative-decoding','Keep only the verified prefix','Where does speculative output stop being valid?','One illustrative four-draft round matching the figure: first three accepted, fourth rejected. Declared conditional p/q probabilities and uniform draws expose the acceptance rule. Target KV appears only when verification consumes the proposals. No measured timing or speedup. Correction uses the normalized positive residual, not an ordinary target sample.','sequence',[seq('draft','Sequential draft proposals',0,4),seq('tests','Target acceptance probability',1,4,'compute'),seq('draws','Uniform draws u',2,4,'reference'),seq('emitted','Committed target-distributed output',3,4,'output'),pool('tentative','Target KV for consumed speculative inputs',4,4),reg('correction','Residual correction at rejection',5,'distribution')],['A proposal is not emitted before target verification and acceptance.','The first rejection ends the valid draft prefix; later drafts would be discarded.','Exact sampling requires min(1,p/q) acceptance and normalized positive-residual correction.'])
 st={'draft':s([None]*4),'tests':s([None]*4),'draws':s(uniforms),'emitted':s([None]*4),'tentative':p([None]*4),'correction':r(None)}
 def sm():return [metric('proposed','Draft positions',sum(x is not None for x in st['draft']['values']),'tokens'),metric('emitted','Committed output',sum(x is not None for x in st['emitted']['values']),'tokens')]
 step(L,'start','Retain context before proposing','The already accepted context is outside this suffix view; no speculative token is committed.',st,sm(),'All proposed output slots are unwritten.')
 for j,v in enumerate(draft):
  st['draft']['values'][j]=v
  step(L,'draft'+str(j),'Draft proposal '+v,'The draft depends on preceding proposals; target acceptance has not been tested.',st,sm(),'The proposed prefix grows by one; emitted output remains unchanged.',trans('execute',['draft'],'One sequential draft-model step; a proposal has no target K/V yet.'))
 st['tentative']=p([slot(v+' trial','Request A') for v in draft],'Target K/V: trial = tentative; kept = accepted prefix')
 st['tests']=s([str(pv)+' / '+str(qv)+' → '+str(a) for pv,qv,a in zip(pvals,qvals,accept)],'One target verification pass; a = min(1,p/q)')
 step(L,'verify','Verify all proposed positions','The target consumes the drafted inputs in one pass, producing tentative target K/V and the conditional acceptance probabilities.',st,sm(),'The declared draws can now be compared against each acceptance probability.',trans('execute',['tests','tentative'],'One target forward verification consumes proposals and produces target K/V.'))
 st['emitted']['values'][:accepted]=draft[:accepted];st['tentative']['slots']=[slot(v+' kept','Request A') if i<accepted else x for i,(v,x) in enumerate(zip(draft,st['tentative']['slots']))]
 step(L,'prefix','Commit the accepted prefix','Acceptance proceeds in order and stops at the first u greater than its acceptance probability.',st,sm(),'The first '+str(accepted)+' drafts are committed; T4 is rejected.',trans('write',['emitted','tentative'],'Commit only the verified accepted prefix.'))
 # At rejection p(T4)=.1 q(T4)=.5; other token probabilities .9/.5 produce residual .4 => corrected alternative with probability one.
 residual=[max(pv-qv,0) for pv,qv in zip([.1,.9],[.5,.5])];z=sum(residual);correction=[v/z for v in residual]
 st['correction']=r('T4′ probability '+str(correction[1]),'Normalized positive residual: T4 has probability '+str(correction[0]))
 st['emitted']['values'][accepted]='T4′'
 step(L,'correct','Sample and emit the residual correction','At the first rejection, this two-token conditional example puts all positive residual mass on T4′.',st,sm(),'T4′ replaces T4; the output has the same accepted prefix.',trans('transform',['correction','emitted'],'Sample the normalized positive residual.'))
 st['tentative']['slots'][accepted:]=[None]*(len(draft)-accepted)
 step(L,'rollback','Discard the rejected speculative cache suffix','Tentative state for rejected T4 cannot be reused. The corrected token must be consumed in a subsequent forward pass before its K/V is cached.',st,sm(),'Only the accepted speculative cache prefix remains valid.',trans('release',['tentative'],'Roll back cache at the rejection boundary.'))
 L['details']={'summary':'Exact sampling rule for this declared trace','paragraphs':['At each position, p and q are conditional probabilities of that proposed token under the target and draft. The other tokens carry the remaining probability mass. The visible uniform draw accepts when u is no greater than min(1,p/q).','At rejection, the two-token target distribution is '+str([.1,.9])+' and the draft distribution is '+str([.5,.5])+'. Positive residual masses '+str(residual)+' normalize to '+str(correction)+'. The corrected alternative therefore has probability '+str(correction[1])+' in this toy conditional. No distributional or latency result is inferred from the observed acceptance count alone.']}
 lessons.append(L)
 # FedAvg scalar toy, explicit selection, completion, upload, aggregation, publication.
 sizes=[2,4,1,3,5];models=[1,3,2,4,9];finishes=[3,4,2,4,7];deadline=5
 eligible=[i for i,t in enumerate(finishes) if t<=deadline];total=sum(sizes[i] for i in eligible);weighted=sum(sizes[i]*models[i] for i in eligible)/total
 L=lesson('fig-federated-averaging-cycle','A deadline changes the averaging population','Which local examples enter the next global model?','Five clients and one scalar model coordinate. Local sample counts, completed scalar models, and finish times are declared toy inputs. Four clients complete before deadline 5. Sample counts represent retained local data; no raw examples upload.','communication',[seq('data','Client-local sample counts (stay local)',0,5,'memory'),seq('client','Client model coordinate',1,5,'compute'),seq('upload','Accepted uploaded model coordinate',2,5,'communication'),reg('global','Published global coordinate',3),reg('aggregate','Round aggregate',4),obj('tasks','lanes','Local work and deadline', 'compute',{'lanes':['C1','C2','C3','C4','C5'],'start':0,'end':8,'axisLabel':'Toy round time; deadline 5','ticks':[0,5,8]},5)],['The denominator includes only clients whose updates are accepted this round.','Local sample counts remain on their clients; model coordinates are uploaded.','An aggregate becomes the global model only at publication.','Keeping raw data local alone does not establish privacy.'])
 for o in L['objects']:
  if o['id'] in ['data','client','upload']:o['config']['slotLabels']=['C'+str(i+1) for i in range(len(sizes))]
 segments=[dict(id='client'+str(i),lane=i,start=0,end=t,label='local train',state='waiting',role='compute') for i,t in enumerate(finishes)]
 st={'data':s(sizes),'client':s([None]*5),'upload':s([None]*5),'global':r(0),'aggregate':r(None),'tasks':dict(segments=segments,cursor=0,status='Cohort selected; deadline 5')}
 def fm():return [metric('accepted','Accepted clients',sum(x is not None for x in st['upload']['values']),'clients'),metric('examples','Accepted local examples',sum(sizes[i] for i,x in enumerate(st['upload']['values']) if x is not None),'examples')]
 step(L,'selected','Select the five-client cohort','The coordinator chooses the round population; no local model is initialized yet.',st,fm(),'All raw examples remain local.')
 st['client']=s([st['global']['value']]*len(sizes),'Copies of global version t');st['tasks']['segments']=[dict(x,state='running') for x in segments]
 toks=[token('broadcast'+str(i),0,at('client',i),owner='Client '+str(i+1),state='copy',source=at('global')) for i in range(len(sizes))]
 step(L,'broadcast','Broadcast the current global model','Each client receives a model copy and begins its independent local work.',st,fm(),'Clients begin from the same global coordinate.',trans('copy',['global'],'Broadcast copies without moving the server model.',at('global'),at('client')),toks)
 st['client']['values']=[models[i] if i in eligible else 0 for i in range(len(sizes))];st['client']['status']='Local updates completed except C5, still unfinished';st['tasks']['segments']=[dict(x,state='done' if i in eligible else 'running') for i,x in enumerate(segments)];st['tasks']['cursor']=deadline
 step(L,'local','Advance local work to the round deadline','The four on-time clients complete; C5 is still running beyond the deadline.',st,fm(),'Local updates exist, but the server has accepted none yet.',trans('execute',['client','tasks'],'Complete the parallel local work eligible by the deadline.'))
 st['upload']=s([models[i] if i in eligible else None for i in range(len(sizes))],'C5 excluded at deadline');st['tasks']['segments'][-1]['state']='blocked'
 toks=[token('upload'+str(i),models[i],at('upload',i),owner='Client '+str(i+1),state='copy',source=at('client',i)) for i in eligible]
 step(L,'upload','Accept on-time uploads','C5 contributes neither a model nor examples to this round’s denominator.',st,fm(),'Only accepted clients define the aggregation population.',trans('copy',['client'],'Upload completed model coordinates.',at('client'),at('upload')),toks)
 st['aggregate']=r(weighted,'Computed; publication pending',expression='('+' + '.join(str(sizes[i])+' × '+str(models[i]) for i in eligible)+') / '+str(total))
 step(L,'average','Compute the example-weighted aggregate','Each accepted coordinate is weighted by its client’s local sample count.',st,fm(),'The unpublished aggregate is '+str(weighted)+'; global version t is still active.',trans('execute',['aggregate'],'Compute weighted average of accepted updates.'))
 st['global']=r(weighted,'Published version t+1');st['aggregate']['status']='Published as global version t+1'
 step(L,'publish','Publish the next global version','The completed aggregate becomes the next round’s starting model.',st,fm(),'The global model changes only after aggregation completes.',trans('write',['global'],'Publish the aggregate.',at('aggregate'),at('global')))
 L['checks']=[{'kind':'constantValues','object':'data'}];lessons.append(L)
 # Frozen linear + ReLU + scalar classifier: precise minimal bias-only subcase.
 x=-1;w=2;b=3;c=2;lr=Fraction(1,10);pre=w*(x+b);mask=int(pre>0);h=max(pre,0);y=c*h;target=0;dy=y-target;dc=dy*h;dh=dy*c;dpre=dh*mask;db=dpre*w;newb=float(b-lr*db);newc=float(c-lr*dc)
 L=lesson('fig-tinytl-architecture','Frozen weights still carry the backward path','Why must a frozen backbone still execute backward computation?','Focused bias-only subcase of TinyTL: trainable input bias b=3, frozen scalar linear weight w=2, ReLU, trainable scalar classifier c=2, input −1, target 0, half-squared loss, SGD rate 0.1. Lite residuals and normalization are omitted explicitly; no claim about full TinyTL memory savings.','ownership',[seq('parameters','Parameters: frozen w, trainable b, c',0,3,'memory'),chamber('forward','Forward values: preactivation, h, y',1,3),pool('saved','Live backward state: ReLU mask, classifier input',2,2),chamber('grad','Gradients: dc, dh, dpre, db',3,4),seq('input','Original input and target',4,2,'reference')],['The frozen weight w never receives an optimizer update.','ReLU backward requires its mask even though w is frozen.','The classifier input stays live until dc is computed.','Removing a weight gradient does not remove the gradient path to an earlier bias.'])
 st={'parameters':s([w,b,c]),'forward':s([None]*3),'saved':p([None]*2),'grad':s([None]*4),'input':s([x,target])}
 def tm():return [metric('live','Retained backward items',count(st['saved']['slots']),'items'),metric('trainable','Trainable scalar parameters',len([b,c]),'parameters')]
 step(L,'ready','Start with a frozen backbone weight','The bias and classifier remain trainable. This single linear block needs no saved input for a weight gradient.',st,tm(),'Freezing w removes its optimizer update, not its computation.')
 st['forward']=s([pre,h,y],'Forward completed');st['saved']=p([slot(mask,'training step'),slot(h,'training step')])
 step(L,'forward','Execute forward and retain required backward state','The ReLU mask supports db; the classifier input supports dc.',st,tm(),'The necessary backward state remains live after forward.',trans('execute',['forward','saved'],'Compute forward values and retain last-consumer state.'))
 st['grad']['values'][:2]=[dc,dh]
 step(L,'classifier','Backpropagate through the classifier','The classifier consumes retained h for dc and sends dh toward the frozen block.',st,tm(),'The hidden-state gradient exists even though w is frozen.',trans('execute',['grad'],'Compute dc and dh from the loss gradient.'))
 st['saved']['slots'][1]=None
 step(L,'releaseh','Release the classifier input after its last use','The classifier gradient is complete; h is no longer needed for backward.',st,tm(),'The ReLU mask stays retained for the earlier bias gradient.',trans('release',['saved'],'Release h after dc.'))
 st['grad']['values'][2]=dpre
 step(L,'relu','Backpropagate through ReLU','The retained mask selects the preactivation gradient; the earlier bias still needs a gradient through frozen w.',st,tm(),'The mask has served its last consumer.',trans('execute',['grad'],'Compute dpre using the retained mask.'))
 st['saved']['slots'][0]=None
 step(L,'releasemask','Release the consumed ReLU mask','All backward consumers of the saved mask have finished.',st,tm(),'No saved backward items remain.',trans('release',['saved'],'Release mask after dpre.'))
 st['grad']['values'][3]=db
 step(L,'backbone','Backpropagate through the frozen linear weight','The frozen weight still multiplies the incoming gradient to reach the trainable input bias; dw is omitted.',st,tm(),'db = dpre × w; freezing w does not cut this backward path.',trans('execute',['grad'],'Compute db through frozen w.'))
 st['parameters']=s([w,newb,newc],'SGD updates only b and c')
 step(L,'update','Update only the trainable parameters','SGD subtracts learning rate times gradient from b and c; w stays fixed.',st,tm(),'w remains '+str(w)+'; b becomes '+str(newb)+' and c becomes '+str(newc)+'.',trans('write',['parameters'],'Apply trainable-only SGD.'))
 L['checks']=[{'kind':'metricEqualsOccupied','metric':'live','object':'saved'},{'kind':'constantValues','object':'input'}];lessons.append(L)
 # Shadow/canary use same physical surfaces but explicit alternative cases.
 objs=[seq('requests','Request IDs: original traffic',0,4),seq('production','Production v1 execution',1,4,'compute'),seq('candidate','Candidate v2 execution',2,4,'compute'),seq('users','User-visible response provenance',3,4,'output'),seq('comparison','Offline paired comparison',4,4,'control'),reg('gate','Rollout gate',5,'decision')]
 L=lesson('fig-shadow-deployment','Copies versus exposed traffic','Which execution is allowed to answer each user?','Four illustrative request IDs. Original case mirrors every request in shadow mode. The alternative case routes one of four requests to a canary, illustrating the 25% stage listed in the static figure. No quality conclusion follows from these synthetic outputs.','communication',objs,['Shadow requests are copies with stable original request IDs.','Each user receives exactly one model’s response.','Canary routing partitions traffic rather than mirroring it.','Synthetic evidence cannot authorize promotion; a failed gate can stop further exposure.'])
 def shadow_steps(canary=False):
  T=copy.deepcopy(L);T['steps']=[];req=['R'+str(i+1) for i in range(4)];chosen=len(req)-1;candidate_ids=[chosen] if canary else list(range(len(req)));prod_ids=[i for i in range(len(req)) if not canary or i!=chosen]
  st={'requests':s(req),'production':s([None]*4),'candidate':s([None]*4),'users':s([None]*4),'comparison':s([None]*4),'gate':r('Await evidence')}
  def mm():return [metric('executions','Completed model executions',sum(isinstance(v,str) and (v.startswith('v1:') or v.startswith('v2:')) for k in ['production','candidate'] for v in st[k]['values']),'executions'),metric('exposed','Candidate user responses',sum(v=='v2' for v in st['users']['values']),'responses')]
  step(T,'arrive','Receive the request batch','Request IDs retain identity across execution and response provenance.',st,mm(),'No model output is available yet.')
  st['production']['values']=[req[i] if i in prod_ids else None for i in range(len(req))];st['candidate']['values']=[req[i] if i in candidate_ids else None for i in range(len(req))]
  toks=[]
  if not canary:
   toks=[token('shadow'+str(i),req[i],at('candidate',i),owner=req[i],state='copy',source=at('requests',i)) for i in candidate_ids]
  step(T,'route','Partition requests for the canary' if canary else 'Mirror each request to the shadow candidate','The splitter chooses one execution path per request.' if canary else 'Production and candidate both receive the same request IDs; shadow copies cannot answer users.',st,mm(),'Each request has one execution assignment.' if canary else 'Each request has two execution assignments, with candidate exposure still zero.',trans('write' if canary else 'copy',['production','candidate'] if canary else ['requests'],'Assign one path.' if canary else 'Make candidate copies with provenance.',at('requests'),at('candidate')),toks)
  st['production']['values']=['v1:'+req[i] if i in prod_ids else None for i in range(len(req))];st['candidate']['values']=['v2:'+req[i] if i in candidate_ids else None for i in range(len(req))]
  step(T,'execute','Execute the assigned model calls','Both model surfaces show completed outputs; only the routing policy can select user responses.',st,mm(),'Candidate outputs are available, but none has yet reached a user.',trans('execute',['production','candidate'],'Complete independent assigned inference calls.'))
  st['users']=s(['v2' if canary and i in candidate_ids else 'v1' for i in range(len(req))],'Exactly one response per request')
  step(T,'reply','Return the authorized response per request','Canary responses expose the selected users to v2.' if canary else 'The shadow candidate stays outside the response path; all users receive v1.',st,mm(),'Exposure follows the response route, not the number of executions.',trans('write',['users'],'Commit exactly one authorized response, with v1/v2 provenance per request.',to=at('users')))
  if canary:
   st['comparison']['values'][chosen]='SLO FAIL';st['gate']=r('Production only: STOP v2','Synthetic SLO gate failed; future candidate share is zero')
   step(T,'gate','Apply the declared failed canary gate','The illustrative canary output breaches a prespecified SLO gate. Further candidate exposure stops; past responses cannot be recalled.',st,mm(),'One user was exposed; the next request must use production.',trans('write',['comparison','gate'],'Record gate failure and block further rollout.'))
  else:
   st['comparison']=s(['v1/v2:'+q for q in req],'Paired outputs; quality requires labels or oracle')
   step(T,'compare','Copy paired outputs to offline comparison','The comparison matches outputs by request ID. Differences alone do not establish quality improvement.',st,mm(),'Every request is paired offline; no user saw a candidate output.',trans('write',['comparison'],'Store paired comparison evidence.',at('candidate'),at('comparison')))
  return T['steps']
 L['steps']=shadow_steps();L['variants']=[dict(id='canary',label='Canary: one of four',scope='Same four toy requests; the splitter sends one selected request to v2 and the rest to v1, illustrating the listed 25% canary stage. The synthetic gate fails and rolls back future candidate traffic.',steps=shadow_steps(True))];lessons.append(L)
 # Historical availability, not just database validity.
 times=[1,4,6,10,13,15,17,19,21,23];cutoff=12;early=[t for t in times if t<=cutoff];late=[t for t in times if t>cutoff]
 L=lesson('fig-time-travel','Keep the noon feature frozen','Why is a correct midnight count wrong for a noon example?','The caption’s click trace: four events available before noon and six later events. Event time equals availability time in this toy; no late arrivals. The noon training row stays frozen while the event log advances to midnight.','sequence',[seq('events','Append-only click log: hour of day',0,len(times)),reg('clock','Current data availability time',1,'hour','hour'),reg('noon','Frozen noon feature',2,'count','clicks'),reg('midnight','Current live count',3,'count','clicks'),reg('legal','Point-in-time joined feature',4,'count','clicks'),reg('illegal','Naive midnight joined feature',5,'count','clicks')],['A noon feature includes only events available by noon.','Later data can change the live count without changing a historical feature.','Event timestamps alone are insufficient if actual data availability is later.'])
 st={'events':s([None]*len(times)),'clock':r(0),'noon':r(None),'midnight':r(0),'legal':r(None),'illegal':r(None)}
 def ptm():return [metric('logged','Available click events',sum(t is not None for t in st['events']['values']),'events'),metric('future','Future events in frozen noon value',0 if st['noon']['value'] is not None else 'pending','events')]
 step(L,'empty','Start the daily log','No clicks have arrived and the noon feature has not been frozen.',st,ptm(),'Unwritten feature values remain distinct from a zero count.')
 st['events']['values'][:len(early)]=early;st['clock']=r(cutoff);st['midnight']=r(len(early))
 step(L,'early','Append the four pre-noon clicks','This parallel append batches only events whose declared availability precedes the prediction cutoff.',st,ptm(),'The live count at noon includes exactly the early clicks.',trans('write',['events','clock','midnight'],'Append available early events.'))
 st['noon']=r(len(early),'As available at noon')
 step(L,'freeze','Freeze the noon feature','The historical training row records the count available at its prediction moment.',st,ptm(),'The historical value remains fixed as the log grows.',trans('write',['noon'],'Freeze the legal historical count.',at('midnight'),at('noon')))
 st['events']['values']=times[:];st['clock']=r(24);st['midnight']=r(len(times))
 step(L,'late','Append the six later clicks','Later arrivals update the current count while leaving the noon row unchanged.',st,ptm(),'The database now has a valid midnight total that was unavailable at noon.',trans('write',['events','clock','midnight'],'Append post-cutoff events.'))
 st['legal']=r(len(early),'Join with historical availability boundary')
 step(L,'legal','Join using the noon availability boundary','A point-in-time lookup retrieves the frozen historical feature.',st,ptm(),'The noon training example sees '+str(len(early))+' clicks.',trans('write',['legal'],'Read the legal historical value.',at('noon'),at('legal')))
 st['illegal']=r(len(times),'LEAKAGE: reads midnight state')
 step(L,'illegal','Compare the naive midnight join','The naive lookup substitutes later state into an earlier training row.',st,ptm(),'The naive feature leaks '+str(len(late))+' future events and inflates the value by '+str(len(times)/len(early))+'×.',trans('write',['illegal'],'Expose the illegal future-bearing feature.',at('midnight'),at('illegal')))
 lessons.append(L)
 # Explicit physical toy inputs, no hardware registry, simple piecewise-constant marginal trace.
 release=0;start=2;duration=2;deadline=5;power=1;high=600;low=200;overheads={'checkpoint':30,'idle':40,'migration':20};baseline=duration*power*high;operational=duration*power*low;net=operational+sum(overheads.values());saving=baseline-net
 L=lesson('fig-carbon-aware-scheduling','Feasibility before carbon savings','When does waiting produce a feasible net emissions reduction?','Synthetic marginal grid factors: 600 gCO₂e/kWh during hours 0–2, 200 during 2–5. Same job uses 1 kW for two hours. Dedicated baseline capacity is available immediately; shifted capacity is busy until hour 2. Deadline hour 5. Declared checkpoint/idle/migration charges are toy accounting inputs, not measured infrastructure.','schedule',[obj('grid','plot','Synthetic marginal grid signal','reference',{'x':{'min':0,'max':5,'label':'Hours','ticks':[0,2,4,5]},'y':{'min':0,'max':600,'label':'gCO₂e/kWh','ticks':[0,200,600]}},0),obj('schedule','lanes','Job and capacity constraints','control',{'lanes':['Immediate baseline','Shifted job','Shifted headroom'],'start':0,'end':5,'axisLabel':'Hours; deadline 5','ticks':[0,2,4,5]},1),reg('base','Feasible immediate baseline emissions',2,'carbon','gCO₂e'),reg('run','Shifted execution emissions',3,'carbon','gCO₂e'),seq('charges','Charges: checkpoint, idle, migration',4,3,'memory'),reg('net','Shifted net emissions',5,'carbon','gCO₂e')],['Both schedules perform the same useful work before the deadline.','A low grid signal alone does not establish available capacity or enough runtime.','Net emissions include checkpoint, idle-wait, and migration charges.','The synthetic marginal signal is separate from the chapter’s measured European energy chart.'])
 seg=[dict(id='baseline',lane=0,start=release,end=duration,label='same work',state='done',role='compute'),dict(id='shift',lane=1,start=start,end=start+duration,label='same work',state='waiting',role='compute'),dict(id='busy',lane=2,start=0,end=start,label='capacity busy',state='blocked',role='control'),dict(id='headroom',lane=2,start=start,end=deadline,label='headroom available',state='waiting',role='memory')]
 st={'grid':dict(series=[dict(id='signal',label='Synthetic marginal signal',role='reference',points=[[0,high],[2,high],[2,low],[5,low]])],markers=[],status='Piecewise-constant synthetic signal'),'schedule':dict(segments=seg,cursor=0,status='Checkpointed job can wait'),'base':r(baseline),'run':r(0),'charges':s([None]*3),'net':r(None)}
 def cm():return [metric('finished','Shifted useful runtime completed',sum(x['end']-x['start'] for x in st['schedule']['segments'] if x['id']=='shift' and x['state']=='done'),'hours'),metric('slack','Finish-to-deadline slack',deadline-(start+duration),'hours')]
 step(L,'wait','Hold the job checkpointed','The immediate baseline is feasible on separate dedicated capacity. The shifted pool is busy during the high-carbon interval.',st,cm(),'Waiting is a candidate schedule; its net benefit is not yet established.')
 st['schedule']['status']='Job running in feasible low-carbon window';st['schedule']['cursor']=start;st['schedule']['segments'][1]['state']='running';st['schedule']['segments'][2]['state']='done';st['schedule']['segments'][3]['state']='running'
 step(L,'admit','Admit the job in the feasible low-carbon window','The interval has headroom and enough time to finish before the deadline.',st,cm(),'The job can execute within the low-carbon window.',trans('execute',['shift','busy','headroom'],'Evaluate capacity/deadline constraints and begin the bounded task.'))
 st['schedule']['status']='Job completed before the deadline';st['schedule']['cursor']=start+duration;st['schedule']['segments'][1]['state']='done';st['run']=r(operational,expression=str(power)+' kW × '+str(duration)+' h × '+str(low)+' gCO₂e/kWh')
 step(L,'run','Complete the same useful work','The job consumes the same declared execution energy under the lower marginal factor.',st,cm(),'Execution emissions alone omit the cost of shifting.',trans('execute',['shift','run'],'Complete the job and integrate execution emissions.'))
 st['charges']=s(list(overheads.values()),'Declared extra shift charges')
 step(L,'charge','Account for all shifting overheads','Checkpoint I/O, idle standby, and migration add emissions beyond execution.',st,cm(),'The extra charges total '+str(sum(overheads.values()))+' gCO₂e.',trans('write',['charges'],'Enter all declared overhead charges.'))
 st['net']=r(net,expression=str(operational)+' + '+' + '.join(map(str,overheads.values())))
 step(L,'net','Compute net emissions and compare the feasible baseline','The complete accounting adds every shift charge before comparison.',st,cm(),'The shift emits '+str(net)+' gCO₂e versus '+str(baseline)+' gCO₂e, saving '+str(saving)+' gCO₂e under these inputs.',trans('execute',['net'],'Sum execution and shifting terms.'))
 assert start+duration<=deadline and net==operational+sum(overheads.values());lessons.append(L)
 # Delayed labels: four bounded decisions; demonstrate insufficient subgroup support.
 decisions=[('A',1),('A',0),('B',1),('B',1)];labels=[1,1,0,1];min_support=3
 L=lesson('fig-responsible-ai-architecture','Prediction evidence precedes outcome evidence','Which monitoring metrics can exist before outcomes arrive?','Four bounded toy decisions: two per group. Labels arrive later and are joined by decision ID. Descriptive rates are not reliable fairness judgments; the declared subgroup minimum is three labeled decisions. Model v1 remains serving while an owner investigates.','control',[seq('decisions','Bounded decision log: ID / group / prediction',0,4,'memory'),seq('labels','Delayed labels matched by decision ID',1,4,'input'),seq('parity','Positive predictions / group decisions: A, B',2,2,'output'),seq('errors','Errors / labeled decisions: A, B',3,2,'output'),reg('serving','Synchronous serving version',4,'model version'),pool('review','Asynchronous investigation and release authority',5,2)],['Prediction-only counts do not require labels.','Error rates require matched outcomes for the same decision IDs.','Small subgroup windows remain marked as insufficient evidence.','A monitoring alarm neither proves unfairness nor authorizes a model change.'])
 st={'decisions':s([None]*4),'labels':s([None]*4),'parity':s([None]*2),'errors':s([None]*2),'serving':r('v1'),'review':p([None]*2)}
 def rm():return [metric('decisions','Logged decisions',sum(x is not None for x in st['decisions']['values']),'decisions'),metric('labels','Matched outcomes',sum(x is not None for x in st['labels']['values']),'labels'),metric('minimum','Declared minimum per subgroup',min_support,'labels/group')]
 step(L,'ready','Keep serving on the synchronous path','Expensive monitoring runs asynchronously; no model change is pending.',st,rm(),'No decision evidence or labels exist yet.')
 st['decisions']=s(['D'+str(i)+': '+g+' / '+str(pred) for i,(g,pred) in enumerate(decisions)],'Bounded evidence from completed synchronous responses')
 step(L,'serve','Serve and append bounded decision records','The request path returns v1 decisions and records IDs, authorized group fields, and predictions for asynchronous analysis.',st,rm(),'Predictions are available while every outcome remains pending.',trans('write',['decisions'],'Append completed decision evidence.'))
 groups=['A','B'];totals=[sum(g==G for g,pred in decisions) for G in groups];positives=[sum(pred for g,pred in decisions if g==G) for G in groups]
 st['parity']=s([str(a)+' / '+str(n) for a,n in zip(positives,totals)],'Prediction-only descriptive counts')
 step(L,'prediction','Update prediction-only subgroup counts','The monitor counts positive decisions by subgroup without guessing outcomes.',st,rm(),'Positive-rate differences are observable; error rates are still unavailable.',trans('execute',['parity'],'Compute prediction-only counts.'))
 st['labels']=s(['D'+str(i)+': '+str(y) for i,y in enumerate(labels)],'Outcome labels joined to old decision IDs')
 step(L,'labels','Match delayed labels to historical decisions','Label arrival changes what evidence is available without changing the old decisions.',st,rm(),'Every error comparison now refers to a matched decision and outcome.',trans('write',['labels'],'Ingest matched delayed outcomes.'))
 errors=[sum(pred!=labels[i] for i,(g,pred) in enumerate(decisions) if g==G) for G in groups]
 st['errors']=s([str(e)+' / '+str(n)+'; insufficient' if n<min_support else str(e)+' / '+str(n) for e,n in zip(errors,totals)],'Descriptive errors; subgroup support below policy minimum')
 step(L,'errors','Compute supported descriptive error counts','The toy counts are computable, but each subgroup remains below the declared support threshold.',st,rm(),'The monitor reports evidence gaps instead of a reliable fairness verdict.',trans('execute',['errors'],'Compute label-dependent counts and check support.'))
 st['review']=p([slot('Investigate support gap','Audit owner','ready'),slot('Release approval pending','Release owner','allocated')])
 step(L,'investigate','Open an owner-governed investigation','The asynchronous owner reviews evidence coverage. Validation and explicit approval are still required for any release.',st,rm(),'v1 stays serving; the dashboard has no authority to publish a replacement.',trans('write',['review'],'Create investigation and pending approval records.'))
 L['checks']=[{'kind':'constantValues','object':'serving'}];lessons.append(L)
 return lessons

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--write',action='store_true');parser.add_argument('--path',type=Path,default=Path(__file__).with_name('vol2-c.json'));args=parser.parse_args();expected=build()
 if args.write:args.path.write_text(json.dumps(expected,ensure_ascii=False,indent=2)+'\n')
 actual=json.loads(args.path.read_text());assert actual==expected,'JSON differs from independently rerun declared-input calculations; inspect before using --write.'
 assert len({L['key'] for L in actual})==len(actual)
 for L in actual:
  for name,steps in [('original',L['steps'])]+[(v['id'],v['steps']) for v in L.get('variants',[])]:
   for S in steps:
    print(L['key'],name,S['id'],json.dumps({m['id']:m['value'] for m in S['snapshot']['metrics']},ensure_ascii=False))
 print('PASS: every snapshot and declared variant matches recomputed scalar, count, occupancy, weighted-average, residual-probability, timing, and emissions endpoints.')
if __name__=='__main__':main()

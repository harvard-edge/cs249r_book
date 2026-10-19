#!/usr/bin/env python3
"""Recompute the explicitly scoped illustrative cases; default verifies without writes."""
import copy
import json
import math
from pathlib import Path
import sys

def obj(id,kind,label,config,n,role='reference'):
 return dict(id=id,kind=kind,label=label,role=role,config=config,layout={'wide':{'row':n//2,'column':n%2},'narrow':{'row':n,'column':0}})
def axis(a,b,label): return dict(min=a,max=b,label=label)
def plot(id,label,x,y,n): return obj(id,'plot',label,dict(x=x,y=y),n)
def seq(id,label,slots,n): return obj(id,'sequence',label,dict(slots=slots),n)
def nl(id,label,a,b,n): return obj(id,'numberLine',label,dict(min=a,max=b,axisLabel=label),n)
def series(id,label,pts,role='reference'): return dict(id=id,label=label,points=pts,role=role)
def marker(id,x,y,label): return dict(id=id,x=x,y=y,label=label)
def ps(ss,ms=(),status=''): return dict(series=ss,markers=list(ms),status=status)
def vs(vals,status=''): return dict(values=vals,status=status)
def ns(ms,status='',levels=()): return dict(levels=list(levels),markers=ms,status=status)
def nm(id,value,label,role='output'):return dict(id=id,value=value,label=label,role=role)
def met(id,value,unit='',label=None): return dict(id=id,label=label or id.replace('-',' ').capitalize(),value=value,unit=unit)
def step(id,title,explanation,states,result,metrics=(),kind=None,subjects=(),tokens=()):
 return dict(id=id,title=title,explanation=explanation,snapshot=dict(objects=copy.deepcopy(states),tokens=list(tokens),metrics=list(metrics),activeConnections=[],result=result),transition=None if kind is None else dict(kind=kind,subjects=list(subjects),meaning=explanation))
def lesson(key,title,q,scope,family,objects,steps,invariants,checks=()):
 return dict(schema=1,key='vol4:'+key,title=title,question=q,scope=scope,family=family,objects=objects,steps=steps,invariants=invariants,checks=list(checks))
def build():
 out=[]
 # One-degree fixed heading: geometric relation from the source, with a separate toy aisle.
 theta=math.radians(1); clearance=.1; end=clearance/math.tan(theta)
 objects=[plot('replay','Recorded-state replay',axis(0,6,'Forward position (toy m)'),axis(0,.12,'Lateral position (toy m)'),0),plot('physical','Admitted fixed-heading rollout',axis(0,6,'Forward position (toy m)'),axis(0,.12,'Lateral position (toy m)'),1),seq('observation','Policy input: recorded / live lateral position',2,2)]
 states={};steps=[];history=[[0,0]];obs=[0,0]
 for i,(x,feedback) in enumerate([(0,True),(2,False),(2,True),(4,False),(4,True),(end,False),(end,True)]):
  y=x*math.tan(theta)
  if not history or history[-1][0]!=x:history.append([x,y])
  if feedback:obs=[0,y]
  states={'replay':ps([series('recorded','Recorded centerline',[[0,0],[6,0]])],[marker('record',x,0,'Recorded input')]),'physical':ps([series('limit','Side clearance',[[0,clearance],[6,clearance]]),series('trajectory','Actual path',history,'output')],[marker('base',x,y,'Actual base')]),'observation':vs(obs,'Recorded state resets; live state persists')}
  title='Begin on the centerline' if i==0 else ('Feed the displaced state back' if feedback else 'Advance the admitted fixed heading')
  expl='Replay supplies the stored centerline; rollout supplies the state produced by the preceding motion.' if feedback else 'The base advances along the same biased heading; its lateral position integrates the admitted motion.'
  steps.append(step('s'+str(i),title,expl,states,'The live observation retains lateral displacement.' if feedback else ('The toy side clearance is exhausted; this lesson admits no further motion.' if i==5 else 'The next recorded state remains on the centerline while the physical base drifts.'),[met('lateral',round(y,6),'toy m'),met('mse',round(theta**2,8),'rad²')],None if i==0 else ('write' if feedback else 'move'),['observation'] if feedback else ['base']))
 out.append(lesson('fig-07-open-vs-closed-loop','Replay resets; physical drift persists','Why can the same small angular error produce different observations?','Illustrative navigation geometry: fixed 1° heading, toy side clearance 0.10 m, forward position rather than elapsed time. Plot axes use different display scales, matching the source trajectory comparison. This is not the running-machine aisle or the latch policy; no recovery controller is modeled. Motion stops at the clearance boundary.','spatial',objects,steps,['Replay inputs remain recorded centerline states independent of predictions.','Physical lateral displacement equals forward displacement times tan(heading).','Holding at the boundary illustrates the end of this geometric example, not a certified stop policy.']))
 # Rolling shutter: five representative row bands, not the actual sensor mode.
 objects=[obj('image','matrix','Readout assembled from different epochs',dict(rows=5,columns=5,rowLabels=['upper','row 1','middle','row 3','lower']),0,'output'),plot('camera','Camera position at each row epoch',axis(0,4,'Row epoch (toy ticks)'),axis(0,4,'Along-travel position (toy units)'),1),seq('epochs','Each row keeps its own capture epoch',5,2),nl('exposure','Current row exposure footprint',-1,5,3)]
 steps=[];vals=[None]*25;epochs=[None]*5;points=[]
 for i in range(6):
  if i:
   row=i-1;epochs[row]=row; vals[row*5+row]='edge'
   if row>0:vals[row*5+row-1]='blur'
   if row<4:vals[row*5+row+1]='blur'
   points.append([row,row])
  states={'image':vs(vals,'Each edge cell is a row sample; adjacent blur is exposure spread'),'camera':ps([series('positions','Row positions',points,'input')] if points else []),'epochs':vs(epochs,'Dating the midpoint does not replace these epochs'),'exposure':ns([] if not i else [nm('left',row-.5,'Exposure start','input'),nm('center',row,'Row midpoint','reference'),nm('right',row+.5,'Exposure end','input')])}
  steps.append(step('s'+str(i),'Await the upper row' if not i else 'Sample '+['upper row','second row','middle row','fourth row','lower row'][row],'One row samples the same stationary upright from the camera position at that row epoch. Exposure spread is separate from displacement between row midpoints.',states,'The upright is assembled from different camera positions; the blur footprint belongs to each exposure.' if i else 'No row has been sampled.',[met('sampled',sum(x is not None for x in epochs),'row bands'),met('skew',0 if not points else points[-1][1]-points[0][1],'toy units'),met('blur',1,'toy units')],None if not i else 'write',['image','epochs','camera','exposure']))
 out.append(lesson('fig-08-rolling-shutter-rows','Rows sample a moving camera at different epochs','Why does a stationary upright become skewed while each row also blurs?','Illustrative five-row-band raster, one position unit per row tick, constant camera translation and one-unit exposure spread. The grid is a schematic of along-travel displacement, not a calibrated camera projection or the chapter sensor mode. No yaw or compensation is simulated.','spatial',objects,steps,['Every written row retains its own capture epoch.','Skew compares row midpoint positions; blur measures motion during one row exposure.','A midpoint frame timestamp does not make the row samples simultaneous.'],[dict(kind='metricEqualsNonNull',metric='sampled',object='epochs'),dict(kind='nondecreasingMetric',metric='sampled')]))
 # Shared physical pages: aliases are references, never payload allocations.
 objects=[obj('pages','pool','Physical storage: capture A / converted B',dict(slots=2,slotLabels=['A','B'],columns=2),0,'memory'),seq('references','Outstanding mappings: CPU / ISP / GPU',3,1),obj('work','lanes','Completion obligations (schematic order)',dict(lanes=['DMA','CPU','ISP','GPU'],start=0,end=5,axisLabel='Logical progress'),2,'compute')]
 specs=[('dma',0,0,1,'Capture write'),('cpu',1,1,2,'Read A'),('isp',2,1,3,'Convert A → B'),('gpu',3,1,4,'Read A')]
 steps=[]
 for i in range(8):
  a=None if i==7 else dict(value='frame F',owner='Capture DMA' if i==0 else ('Consumer references' if 2<=i<6 else 'Capture driver'),status='inflight' if i==0 else ('shared' if 2<=i<6 else 'ready'))
  b=None if i<3 else dict(value='converted F',owner='Downstream consumer',status='retained')
  refs=[None,None,None] if i<2 else ['A:F' if i<4 else None,'A:F' if i<5 else None,'A:F' if i<6 else None]
  segs=[]
  for sid,lane,start,end,label in specs:
   done= i>=1 if sid=='dma' else i>=dict(cpu=4,isp=3,gpu=6)[sid]
   running=(i==0 if sid=='dma' else i>=2)
   segs.append(dict(id=sid,lane=lane,start=start,end=end,label=label,state='done' if done else ('running' if running else 'waiting'),role='communication' if sid=='dma' else 'compute'))
  st={'pages':dict(slots=[a,b],status='A reusable only after every reader releases'),'references':vs(refs,'Aliases identify the same A pages; they are not payload copies'),'work':dict(segments=segs,status='Completion does not itself remove a retained mapping')}
  titles=['DMA owns capture A','Complete the DMA write','Order access and admit mapped readers','Finish ISP conversion into B','CPU releases its A reference','ISP releases its A reference','GPU releases the final A reference','Requeue capture A']
  ex=['The capture engine writes frame F; no consumer may read incomplete capture pages.','DMA completion establishes that the capture write is finished; platform synchronization is still required.','Required device ordering and cache maintenance complete before CPU, ISP, and GPU access the same physical A pages.','The ISP writes a different representation to B while preserving frame identity F and retaining its A reference.','The CPU finishes reading and releases its mapping; other consumers still retain A.','Conversion is complete, so the ISP now releases its A reference; the GPU still retains A.','The final capture-page consumer finishes and releases A. B remains independently owned downstream.','The driver returns A to the capture queue only after completion, synchronization, and all consumer releases.'][i]
  steps.append(step('s'+str(i),titles[i],ex,st,'B remains allocated independently of capture-page reuse.' if i==7 else ('A still cannot be reused while any mapping is retained.' if any(refs) else 'No outstanding reader reference remains.' if i==6 else 'Frame identity alone does not imply identical storage.'),[met('occupied',sum(p is not None for p in [a,b]),'page sets'),met('readers',sum(v is not None for v in refs),'references')],None if i==0 else ('release' if i>=4 else 'execute'),['pages','references','work']))
 out.append(lesson('fig-08-shared-capture-buffer','Share pages; release each consumer','What permits capture-page reuse after zero-copy access?','One schematic frame F, source page set A and an ISP-converted output B. Mappings are aliases, not new payload buffers. Task coordinates encode logical obligations, not durations or simultaneous execution; platform-specific synchronization details are abstracted into an explicit handoff.','ownership',objects,steps,['A survives every non-destructive reader and is requeued only after the final release.','ISP output B preserves frame identity but owns different physical pages.','DMA completion alone does not establish ordered visibility to every consumer.'],[dict(kind='metricEqualsOccupied',metric='occupied',object='pages')]))
 # Free-space claims expire as reachable occupancy grows.
 depth=5;allowance=1;speed=1;horizon=(depth-allowance)/speed
 objects=[nl('geometry','Distance ahead of the stationary hull',0,5,0),plot('claim','Certified depth erodes with evidence age',axis(0,4,'Evidence age (toy time units)'),axis(0,4,'Certified free depth (toy units)'),1),seq('evidence','Immutable evidence: capture / occlusion boundary',2,2)]
 steps=[]
 for i,t in enumerate([0,1,2,horizon]):
  front=depth-speed*t;free=max(0,front-allowance)
  states={'geometry':ns([nm('hull',0,'Hull','reference'),nm('reserve',allowance,'Protective + localization reserve','control'),nm('boundary',depth,'Hidden-person origin','reference'),nm('front',front,'Reachable front','input')]),'claim':ps([series('ahead','Current hull: certified depth',[[0,depth-allowance],[t,free]]),series('abreast','Abreast of rack: zero claim',[[0,0],[horizon,0]])],[marker('age',t,free,'Claim now')]),'evidence':vs(['capture t₀','occlusion at x = 5'],'No new observation has arrived')}
  steps.append(step('s'+str(i),'Capture the visible free region' if i==0 else 'Age the same evidence', 'A hidden person can approach from the occlusion boundary; the stationary hull receives no new observation.',states,'The claim expires. Propagating the old map cannot renew its capture epoch.' if free==0 else 'The reachable set grows while the depth certified free contracts.',[met('age',t,'toy time units'),met('reach',speed*t,'toy distance units'),met('free',free,'toy distance units')],None if i==0 else 'move',['front','age']))
 out.append(lesson('fig-09-rack-end-erosion','Hidden reach consumes visible free depth','How does evidence age spend a spatial clearance claim?','Illustrative one-dimensional projection of the source rack-end geometry: stationary hull at 0, occlusion boundary at 5 distance units, combined reserve 1, hidden-person speed bound 1 distance unit per time unit. This is not the running-machine numerical budget. An approaching hull needs an additional closing-speed term.','spatial',objects,steps,['The hidden side of the rack remains unobserved at every age.','Certified free depth is max(0, initial depth − reserve − speed bound × age).','Extrapolation changes the claim but never refreshes its evidence epoch.'],[dict(kind='constantValues',object='evidence')]))
 # Spatially limited negative evidence; all off-ray epochs persist.
 rows,cols=3,6;old=['?']*(rows*cols);old[cols+2]='O';old[cols+4]='F';old_epochs=[0]*(rows*cols)
 objects=[obj('occupancy','matrix','Cell state: F = free / O = occupied / ? = unknown',dict(rows=rows,columns=cols),0,'memory'),obj('epochs','matrix','Per-cell evidence epoch',dict(rows=rows,columns=cols),1,'memory'),seq('return','Qualified return: capture epoch / hit cell',2,2),obj('publication','register','Map publication epoch',dict(type='epoch',unit='logical epoch'),3,'output')]
 steps=[];occ=old[:];epochs=old_epochs[:]
 for i in range(6):
  if i==2:
   for c in [0,1]:occ[cols+c]='F';epochs[cols+c]=1
  if i==3:occ[cols+2]='F';epochs[cols+2]=1
  if i==4:occ[cols+3]='O';epochs[cols+3]=1
  st={'occupancy':vs(occ,'Only the middle row is traversed'),'epochs':vs(epochs,'Behind-hit and off-ray cells retain their own epochs'),'return':vs([None,None] if i==0 else [1,'middle row, col 3'],'Coverage and geometric error are qualified'),'publication':dict(value=2 if i==5 else 0,status='Publication does not redate evidence')}
  titles=['Inspect the old local map','Receive one qualified range return','Mark the first crossed cells free','Retire the phantom on the ray','Record the occupied hit','Publish the updated map']
  expl=['Old cell states and epochs are explicit; unknown is not a numeric zero occupancy.','The return observes a bounded ray ending at middle-row column 3, captured at epoch 1.','Traversal writes free evidence and epoch 1 only into crossed cells.','The qualified empty ray contradicts the old phantom, so that cell becomes free at epoch 1.','The return cell becomes occupied; the ray gives no negative evidence behind this hit.','A newer publication epoch distributes the map without changing any cell evidence epoch.'][i]
  steps.append(step('s'+str(i),titles[i],expl,st,'The hidden and off-ray cells remain dated epoch 0 even after publication.' if i==5 else 'Only qualified spatial support can renew a cell claim.',[met('renewed',sum(t==1 for t in epochs),'cells'),met('publication',st['publication']['value'],'logical epoch')],None if not i else ('reveal' if i==1 else 'write'),['return'] if i==1 else ['occupancy','epochs','publication']))
 out.append(lesson('fig-09-ray-update-epochs','A ray renews cells, not the whole map','Which cell epochs change after one range return?','Illustrative 3 × 6 local grid with a qualified middle-row ray and logical epochs 0, 1, and 2. Free-cell updates assume the source coverage, resolution, and geometric-error qualifications. This is a cell-state mechanism, not a probabilistic occupancy threshold model.','spatial',objects,steps,['Every off-ray and behind-hit state and evidence epoch remains unchanged.','Unknown cells remain explicitly unknown unless qualified evidence touches them.','Publication time is distinct from cell capture epoch.'],[dict(kind='nondecreasingMetric',metric='renewed')]))
 # Frame staleness: exact toy geometry alongside the source small-angle bound.
 lever=4;trans=.2;angle=.1;initial=.05;tolerance=.4
 objects=[plot('scene','Fixed slot and wrist-frame geometry',axis(-.2,1,'Lateral position (toy units)'),axis(0,4.5,'Height (toy units)'),0),seq('edge','Retained camera-to-slot: lateral / height / epoch',3,1),nl('bound','Conservative lateral error allowance',0,1,2)]
 objects[0]['config']['equalAxes']=True
 steps=[]
 for i in range(5):
  dx=trans if i>=1 else 0;theta=angle if i>=2 else 0
  tip=[dx+lever*math.sin(theta),lever*math.cos(theta)]
  hybrid=[dx+initial*math.cos(theta)+lever*math.sin(theta),lever*math.cos(theta)-initial*math.sin(theta)]
  bound=initial+dx+lever*abs(theta)
  ss=[series('captureArm','Capture-dated arm',[[0,0],[0,lever]]),series('currentArm','Current arm',[[dx,0],tip],'input')]
  marks=[marker('slot',0,lever,'Fixed real slot'),marker('wrist',dx,0,'Current wrist')]
  if i>=3:marks.append(marker('hybrid',*hybrid,'Mixed-epoch estimate'))
  if i>=4:marks.append(marker('dated',initial,lever,'Capture-dated estimate'))
  st={'scene':ps(ss,marks),'edge':vs([initial,lever,'t₀'],'Camera-relative measurement remains at its capture epoch'),'bound':ns([nm('initial',initial,'Initial uncertainty','reference'),nm('limit',tolerance,'Slot tolerance','control'),nm('aged',bound,'Mixed-epoch error bound')])}
  title=['Capture the stationary slot','Translate the wrist','Rotate the wrist','Compose stale edge with current wrist','Compose at the capture epoch'][i]
  expl=['The camera-relative observation is stored with the capture epoch.','Current wrist translation changes while the old camera-relative edge remains retained.','The old edge rotates with the current wrist if application code mixes epochs.','The mixed-epoch chain moves the estimated slot even though the true slot remains fixed.','A capture-dated chain recovers the original estimate with initial uncertainty; it supplies no new observation.'][i]
  steps.append(step('s'+str(i),title,expl,st,'Temporal consistency removes the artificial motion, but does not erase initial uncertainty or renew evidence.' if i==4 else 'The retained observation cannot be silently treated as a current camera measurement.',[met('translation',dx,'toy units'),met('rotation',theta,'rad'),met('bound',round(bound,6),'toy units')],None if i==0 else ('move' if i<3 else 'transform'),['scene','bound']))
 out.append(lesson('fig-frame-staleness-error','A mixed-epoch chain moves a stationary target','Why can valid matrix operations produce a false target position?','Illustrative planar wrist: lever length 4 units, translation 0.2, rotation 0.1 rad, initial uncertainty 0.05 and slot tolerance 0.4 units. Both scene axes use the same distance unit and equal geometric scale. Geometry uses exact rotation; the displayed conservative allowance uses initial + translation + lever × |angle|. No ingestion delay, dynamics, or new evidence is simulated.','spatial',objects,steps,['The real slot and retained capture-time edge never move.','Current wrist pose must not silently redate a camera-relative observation.','Capture-dated composition retains initial uncertainty and does not renew the observation.'],[dict(kind='constantValues',object='edge')]))
 # Timing with distinct expiration and fallback actions; no full-stop claim.
 e0=1;tol=5;v=2;acc=1;untracked=(tol-e0)/v;tracked=math.sqrt(2*(tol-e0)/acc);onset=.5;infer=4;period=5
 times=[0,untracked,untracked+onset,tracked,tracked+onset,infer,period]
 objects=[plot('error','Error grows while the host is unavailable',axis(0,5,'Age since capture (toy time units)'),axis(0,14,'Target error (toy distance units)'),0),obj('clock','lanes','Host computation and independent MCU deadlines',dict(lanes=['Intent host','Untracked MCU','Tracked MCU'],start=0,end=5,axisLabel='Toy time units'),1,'control'),nl('position','Untracked fixed target versus moving mug',0,11,2)]
 steps=[]
 for i,t in enumerate(times):
  def task(sid,lane,start,end,label):return dict(id=sid,lane=lane,start=start,end=end,label=label,state='done' if t>=end else ('running' if t>=start else 'waiting'),role='compute' if lane==0 else 'control')
  segments=[task('inference',0,0,infer,'Inference'),task('uLease',1,0,untracked,'Untracked lease'),task('uOnset',1,untracked,untracked+onset,'Brake onset'),task('tLease',2,0,tracked,'Tracked lease'),task('tOnset',2,tracked,tracked+onset,'Brake onset')]
  pts=[[z/20,e0+v*z/20] for z in range(math.floor(t*20)+1)]
  pts2=[[z/20,e0+acc*(z/20)**2/2] for z in range(math.floor(t*20)+1)]
  for ps_,f in [(pts,lambda x:e0+v*x),(pts2,lambda x:e0+acc*x*x/2)]:
   if ps_[-1][0]!=t:ps_.append([t,f(t)])
  st={'error':ps([series('u','Untracked velocity bound',pts,'output'),series('t','Tracked residual acceleration',pts2,'input'),series('tolerance','Grasp tolerance',[[0,tol],[period,tol]])],[marker('uNow',t,e0+v*t,'Untracked'),marker('tNow',t,e0+acc*t*t/2,'Tracked')]),'clock':dict(segments=segments,cursor=t,status='Expiry starts fallback; onset is not a complete stop'),'position':ns([nm('fixed',0,'Fixed old setpoint','reference'),nm('mug',v*t,'Actual mug','input')])}
  titles=['Capture target; host begins inference','Untracked target lease expires','Untracked brake onset is reached','Tracked target lease expires','Tracked brake onset is reached','Host inference finally completes','Next proposal epoch arrives']
  explanation=['Two alternative operating modes share capture time; tracker renewal is absent during this host outage.','Untracked error reaches tolerance before inference finishes; the MCU starts its bounded fallback path.','The fallback reaches brake onset without waiting for the host; stopping motion remains to be budgeted.','With velocity canceled, the residual-acceleration bound now exhausts the same target tolerance.','The tracked-mode fallback reaches brake onset independently of inference.','Inference completion cannot renew an observation that has already expired.','A later proposal still requires fresh qualified evidence and admission.'][i]
  steps.append(step('s'+str(i),titles[i],explanation,st,'No model completion or confidence value can refresh the capture epoch.',[met('age',round(t,6),'toy time units'),met('untracked',round(e0+v*t,6),'toy distance units'),met('tracked',round(e0+acc*t*t/2,6),'toy distance units')],None if i==0 else 'execute',['clock','error','mug']))
 out.append(lesson('fig-10-mug-lease-timeline','Target expiry outruns deliberation','What happens when evidence expires while the host still computes?','Illustrative timing case: initial error 1, tolerance 5 distance units, belt-speed bound 2, residual slip acceleration 1; brake-onset allowance 0.5, inference 4 and proposal period 5 time units. Tracked and untracked lanes are alternative modes with no observation renewals and zero additional controller reserve. These are not the chapter machine timings.','control',objects,steps,['Untracked error is initial error + speed × age; tracked residual error is initial error + acceleration × age² / 2.','The MCU expiration and fallback do not depend on host inference completing.','Brake onset is not a stopped state; stopping clearance remains necessary.']))
 # Acceleration-bounded intersample geometry in normalized units.
 a=2;dt=1;limit=3;near=2.5;reserve=a*dt/2;safe=limit-reserve
 objects=[plot('acceleration','Bounded acceleration reverses at midpoint',axis(0,1,'Time within interval (toy units)'),axis(-2.5,2.5,'Acceleration (toy units)'),0),plot('velocity','Sample checks and continuous velocity',axis(0,1,'Time within interval (toy units)'),axis(0,4,'Velocity (toy units)'),1),nl('samples','Endpoint values and contracted ceiling',0,4,2)]
 steps=[]
 for i,t in enumerate([0,.25,.5,.75,1]):
  rise=a*min(t,dt-t);vnear=near+rise;vsafe=safe+rise
  ts=[0]+[x for x in [.25,.5,.75,1] if x<=t]
  points=lambda base:[[x,base+a*min(x,dt-x)] for x in ts]
  st={'acceleration':ps([series('command','Acceleration schedule',[[0,a],[.5,a],[.5,-a],[1,-a]],'input')],[marker('phase',t,a if t<.5 else -a,'Current acceleration')]),'velocity':ps([series('limit','Physical velocity limit',[[0,limit],[1,limit]]),series('near','Endpoint-only candidate',points(near),'output'),series('safe','Contracted candidate',points(safe),'control')],[marker('nearNow',t,vnear,'Endpoint-only velocity'),marker('safeNow',t,vsafe,'Contracted velocity')]),'samples':ns([nm('endpoint',near,'Candidate endpoint','output'),nm('contracted',safe,'Contracted sample ceiling','control'),nm('limit',limit,'Physical limit','reference')])}
  steps.append(step('s'+str(i),['Begin at valid sampled velocities','Integrate positive acceleration','Reach the hidden velocity peak','Integrate negative acceleration','Return to valid endpoint samples'][i],'Velocity changes by the signed acceleration area. Equal negative area restores endpoint velocity without undoing the intermediate peak.',st,'Both endpoints pass; the continuous endpoint-only candidate nevertheless crossed the limit.' if i==4 else ('The contracted candidate reaches the limit while the endpoint-only candidate exceeds it.' if i==2 else 'The analytical reserve leaves room for the bounded interior velocity rise.'),[met('time',t,'toy time units'),met('reserve',reserve,'toy velocity units'),met('near',vnear,'toy velocity units'),met('contracted',vsafe,'toy velocity units')],None if i==0 else 'execute',['nearNow','safeNow','phase']))
 out.append(lesson('fig-11-intersample-overshoot','Valid samples can hide an interior peak','Why reserve velocity margin between admission samples?','Normalized illustrative interval: duration 1, acceleration bound 2, velocity limit 3, candidate endpoint 2.5. Acceleration reverses instantaneously at the midpoint; no finite jerk bound is imposed. The contracted endpoint is limit − acceleration × duration / 2. This shows the source assumption, not a physical safety certification.','control',objects,steps,['Integrating acceleration over the first half produces a rise of acceleration bound × interval / 2.','Equal positive and negative acceleration areas restore the same endpoint velocity.','The reserve is justified only for an interpolant obeying the declared acceleration bound.']))
 return out

def verify(path):
 expected=build(); actual=json.loads(path.read_text())
 assert actual==expected, 'Authored JSON differs from recomputed lesson states or numerical derivations'
 for l in actual:
  for s in l['steps']:
   print(l['key']+'/'+s['id']+': '+', '.join(m['id']+'='+str(m['value'])+' '+m['unit'] for m in s['snapshot']['metrics']))
 print('PASS:',len(actual),'lessons; complete states, geometry, counts, bounds and endpoints recomputed from declared inputs.')
if __name__=='__main__':
 path=Path(__file__).with_name('vol4-b.json')
 if '--write' in sys.argv:
  path.write_text(json.dumps(build(),ensure_ascii=False,indent=2)+'\n')
 verify(path)

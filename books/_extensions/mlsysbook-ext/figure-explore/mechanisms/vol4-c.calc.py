#!/usr/bin/env python3
"""Recompute all lesson states from owning LEGO cells; default is read-only."""
import argparse, copy, json, math, pathlib, re, sys
ROOT=pathlib.Path(__file__).resolve()
while not (ROOT/'books/vol4').is_dir():
    if ROOT.parent==ROOT:
        raise RuntimeError('Cannot locate repository root containing books/vol4')
    ROOT=ROOT.parent
sys.path.insert(0,str(ROOT/'mlsysim'))
NS={}
def lego(chapter,name):
    p=ROOT/f'books/vol4/{chapter}/{chapter}.qmd'
    for code in re.findall(r'```\{python\}[^\n]*\n(.*?)\n```',p.read_text(),re.S):
        if f'class {name}:' in code:
            exec(compile(code,str(p),'exec'),NS);return NS[name]
    raise ValueError(name)
B=lego('12_enforcement','EnforcementStoppingBudget')
C=lego('12_enforcement','EnforcementContactForce')
M=lego('13_placement','PlacementMachine')
D=lego('13_placement','PlacementDramTiming')
Q=lego('13_placement','PlacementLoadedRead')
H=lego('13_placement','RejectedPlacementThrottle')
F=lego('15_verification','RackEndVerification')
def n(q,unit):return float(q.to(unit).magnitude)
def r(v):return round(float(v),3)
def obj(id,kind,label,config,row,role='reference',full=True):
    return dict(id=id,kind=kind,label=label,role=role,config=config,layout={'wide':{'row':row,'column':0,'columnSpan':2 if full else 1},'narrow':{'row':row,'column':0,'columnSpan':1}})
def axis(lo,hi,label,ticks):return dict(min=lo,max=hi,label=label,ticks=ticks)
def plot(id,label,x,y,row):return obj(id,'plot',label,dict(x=x,y=y),row)
def line(id,label,lo,hi,ticks,row):return obj(id,'numberLine',label,dict(min=lo,max=hi,ticks=ticks,axisLabel=label),row)
def reg(id,label,unit,row):return obj(id,'register',label,dict(type='Scalar',unit=unit),row)
def ps(series=(),markers=(),status=''):return dict(series=list(series),markers=list(markers),status=status)
def series(id,label,points,role='output'):return dict(id=id,label=label,points=[[r(x),r(y)] for x,y in points],role=role)
def marker(id,x,y,label):return dict(id=id,x=r(x),y=r(y),label=label)
def lm(id,value,label,role='output'):return dict(id=id,value=r(value),label=label,role=role)
def ls(markers,status='',levels=()):return dict(levels=list(levels),markers=markers,status=status)
def rs(value,status='',expression=None):
    d=dict(value=r(value) if isinstance(value,(float,int)) else value,status=status)
    if expression:d['expression']=expression
    return d

def metric(id,label,value,unit=''):return dict(id=id,label=label,value=(int(value) if float(value).is_integer() else round(float(value),2)) if isinstance(value,(float,int)) else value,unit=unit)
def step(id,title,explanation,states,metrics,result,kind='execute',subjects=None,first=False):
    return dict(id=id,title=title,explanation=explanation,snapshot=dict(objects=states,tokens=[],metrics=metrics,activeConnections=[],result=result),transition=None if first else dict(kind=kind,subjects=subjects or [next(iter(states))],meaning=explanation))
def lesson(key,title,question,scope,family,objects,steps,invariants,**kw):return dict(schema=1,key='vol4:'+key,title=title,question=question,scope=scope,family=family,objects=objects,steps=steps,invariants=invariants,**kw)
def seg(id,lane,start,end,label,state='done',role='compute'):return dict(id=id,lane=lane,start=r(start),end=r(end),label=label,state=state,role=role)
def lanes(id,label,names,start,end,ticks,row,unit):return obj(id,'lanes',label,dict(lanes=names,start=start,end=end,ticks=ticks,axisLabel=unit),row)
def lane_state(segments,cursor,status=''):return dict(segments=segments,cursor=r(cursor),status=status)

def nonholonomic():
    # Explicit schematic primitive inputs, not physical dimensions or time claims.
    length,width,offset=0.9,0.6,2.0
    poses=[(0,0,0),(0,0,0),(0,0,math.pi/2),(0,offset,math.pi/2),(0,offset,0)]
    objects=[plot('scene','Axle-center pose and body footprint',axis(-1,1,'x, schematic units',[-1,0,1]),axis(-1,3,'y, schematic units',[-1,0,1,2,3]),0),plot('velocity','Candidate planar velocity',axis(-1,1,'x velocity, schematic',[-1,0,1]),axis(-1,1,'y velocity, schematic',[-1,0,1]),1),reg('heading','Heading','degrees',2)]
    objects[0]['config']['equalAxes']=True
    labels=[('start','Start at the sideways goal offset','The base faces along positive x; the destination is along positive y.','The geometric goal does not specify a feasible instantaneous velocity.'),('reject','Test a sideways velocity','A positive y velocity at zero heading violates the no-slip constraint. The actual pose stays fixed.','The candidate velocity is rejected; no sideways slip occurs.'),('turn','Turn about the axle center','Counter-rotating wheels rotate the base while its axle center stays fixed.','The heading now aligns forward motion with the lateral goal.'),('drive','Drive forward along the new heading','Both wheels roll forward; the axle center moves along positive y.','A lateral displacement has been reached through forward rolling.'),('restore','Turn back at the goal','Counter-rotating wheels restore the original heading without translating the axle center.','The goal pose is reached by three admissible motions, each requiring space and time.')]
    steps=[]
    for i,((x,y,th),(id,title,ex,result)) in enumerate(zip(poses,labels)):
        corners=[(-length/2,-width/2),(length/2,-width/2),(length/2,width/2),(-length/2,width/2),(-length/2,-width/2)]
        poly=[(x+a*math.cos(th)-b*math.sin(th),y+a*math.sin(th)+b*math.cos(th)) for a,b in corners]
        ss=[series('body','Current body outline',poly),series('headingRay','Forward heading',[(x,y),(x+length/2*math.cos(th),y+length/2*math.sin(th))]),series('path','Axle-center path',[(0,0),(0,y)],'reference')]
        vx,vy=(0,1) if i in (1,3) else (0,0)
        slip=vx*math.sin(th)-vy*math.cos(th)
        states={'scene':ps(ss,[marker('goal',0,offset,'Goal axle center')]),'velocity':ps([series('candidate','Rejected candidate' if i==1 else 'Admissible translation',[(0,0),(vx,vy)],'input')]),'heading':rs(math.degrees(th))}
        steps.append(step(id,title,ex,states,[metric('offset','Actual lateral displacement',y,'schematic units'),metric('slip','Candidate no-slip residual',slip,'schematic velocity')],result,subjects=['scene','velocity','heading'],first=i==0))
    return lesson('fig-11-nonholonomic-base','A sideways goal needs a feasible maneuver','Why can the base reach a sideways pose but not slide sideways?', 'Obstacle-free schematic turn-drive-turn example. Body proportions match the print schematic; coordinates are arbitrary length units. Rotation sweeps clearance; no obstacle or collision guarantee is modeled.', 'spatial',objects,steps,['During each turn the axle center remains fixed.','The rejected sideways candidate never moves the actual base.','Admitted translation follows the heading and has zero no-slip residual.','Footprints are endpoint outlines, not a collision-checked swept hull.'])

def latch():
    k,v,trip,limit,response,a,stop=[n(q,u) for q,u in [(C.k_latch,'N/m'),(C.v_latch,'m/s'),(C.f_trip,'N'),(C.f_lim,'N'),(C.t_contact,'s'),(C.a_arm,'m/s^2'),(C.t_arm_stop,'s')]]
    tt=trip/(k*v);tc=tt+response;end=tc+stop
    times=[0,tt,tc,tc+stop/2,end]
    objects=[plot('force','Force during the guarded contact',axis(0,math.ceil(end*1000),'Time since first contact, ms',[0,4,8,12]),axis(0,limit*1.1,'Force, N',[0,trip,limit]),0),line('compression','Plate compression, mm',0,limit/k*1000*1.1,[0,r(trip/k*1000),r(limit/k*1000)],1),reg('speed','Arm approach speed','m/s',2)]
    def force(t):
        tau=max(0,t-tc);return k*v*t if t<=tc else n(C.f_cmd,'N')+k*(v*tau-a*tau*tau/2)
    labels=[('contact','Make first contact','The rigid arm reaches the linear-spring plate at the guarded approach speed.'),('trip','Cross the contact tripwire','The tripwire requests a stop; the arm still has its approach velocity.'),('onset','Complete the loop response','Motion during the declared contact-loop response adds compression before braking starts.'),('brake','Decelerate through half the stop','The arm is slowing, but its remaining forward motion still compresses the plate.'),('rest','Bring the arm to rest','The sized deceleration consumes the remaining compression allowance.')]
    steps=[]
    for i,(t,(id,title,ex)) in enumerate(zip(times,labels)):
        f=force(t);speed=max(0,v-a*max(0,t-tc));points=[(t*j/40*1000,force(t*j/40)) for j in range(41)]
        ss=[series('actual','Guarded stop',points),series('limit','Latch limit',[(0,limit),(end*1000,limit)],'reference'),series('tripwire','Contact tripwire',[(0,trip),(end*1000,trip)],'reference')]
        states={'force':ps(ss,[marker('current',t*1000,f,'Current force')]),'compression':ls([lm('current',f/k*1000,'Actual compression'),lm('maximum',limit/k*1000,'Force-limit compression','reference')]),'speed':rs(speed)}
        steps.append(step(id,title,ex,states,[metric('time','Time since contact',t*1000,'ms'),metric('force','Contact force',f,'N'),metric('margin','Force margin',limit-f,'N')],f'Contact force is {f:.2f} N; {limit-f:.2f} N remains before the latch limit.',subjects=['force','compression','speed'],first=i==0))
    return lesson('fig-10-latch-force-budget','Force keeps rising after the tripwire','Why does the stop need the force at braking onset?', 'Canonical guarded latch approach and linear-spring, rigid-arm model from the owning LEGO calculation. Constant arm deceleration is sized to finish at the latch limit; spring work is conservatively omitted. This is model arithmetic, not a measured force trace.', 'control',objects,steps,['Force equals plate stiffness times indentation.','The tripwire does not set arm velocity to zero.','The response interval and braking interval contribute distinct compression.','The modeled endpoint reaches the limit with zero remaining speed; it is not a deployment safety guarantee.'])

def capture():
    v=n(B.v_aisle,'m/s');age=n(B.t_age,'ms');da=n(B.delta_da,'ms');lease=n(B.t_lease,'ms');tick=n(B.t_tick,'ms');bus=n(B.t_bus,'ms');act=n(B.t_act,'ms')
    durations=[age,da,lease,tick,bus,act];names=['Observation to dispatch','Allocated admission','Chunk lease','Next permission tick','Fieldbus delivery','Actuator onset'];ends=[0]
    for d in durations:ends.append(ends[-1]+d)
    objects=[lanes('timeline','Serial capture-to-onset path',['Perception','Proposal','Permission','Drive'],0,260,[0,50,100,150,200,250],0,'Time since exposure midpoint, ms'),line('travel','Base travel before braking, mm',0,350,[0,100,200,300],1),line('budget','Total stop plus fixed allowances, mm',0,1200,[0,600,1100,1200],2)]
    rows=[0,1,2,2,2,3];latest=n(B.capture_to_onset_max,'ms');suffix=n(B.c2_aisle,'mm')+n(B.delta_loc+B.delta_margin+B.eps_track,'mm');clear=n(B.d_clear,'mm')
    steps=[]
    ids=['capture','dispatch','admission','expiry','tick','bus','onset']
    for i,t in enumerate(ends):
        segments=[seg('stage'+str(j),rows[j],ends[j],ends[j+1],names[j], 'done' if j<i else 'waiting', 'communication' if j in (0,1,4) else 'control') for j in range(len(durations))]
        distance=v*t;total=distance+suffix
        states={'timeline':lane_state(segments,t,'Admission is allocated, not qualified'),'travel':ls([lm('base',distance,'Base travel'),lm('latest',v*latest,'Latest-onset travel','reference')]),'budget':ls([lm('total',total,'Stop endpoint plus allowances'),lm('clearance',clear,'Available clearance','reference')])}
        if i==0:ex='A hazard observation is captured at the exposure midpoint. The base is still moving.';title='Capture the hazard frame'
        else:ex=f'{names[i-1]} completes. The base continues at the aisle speed throughout this pre-brake interval.';title='Complete '+names[i-1].lower()
        steps.append(step(ids[i],title,ex,states,[metric('elapsed','Capture-to-current time',t,'ms'),metric('travel','Pre-brake travel',distance,'mm'),metric('reserve','Clearance after modeled stop',clear-total,'mm')],f'The base has spent {distance:.2f} mm; {clear-total:.2f} mm remains after reserving the resident stop and fixed allowances.',subjects=['timeline','travel','budget'],first=i==0))
    return lesson('fig-12-capture-to-onset','Delay spends actual clearance','How does each pre-brake interval spend the stopping reserve?', 'Canonical aisle speed and serial capture-to-onset allocations. Admission latency is an illustrative allocation, not a measured bound. The resident jerk-limited stop and fixed margins are reserved separately; this lesson ends at braking onset.', 'schedule',objects,steps,['Travel before onset equals aisle speed times elapsed time.','Dispatch-to-admission is charged once and is not part of the baseline delay subtotal.','The resident braking distance is not replaced by constant-deceleration braking.','Admission, lease expiry, next tick, bus delivery, and physical onset remain distinct endpoints.'])

def queue():
    sense=n(M.sense,'us'); wcet=n(M.wcet,'us');deadline=n(M.deadline,'us');period=n(M.tick,'us');bpi=n(M.b_plus_i,'us');constraint=n(M.constraint,'us');write=n(M.write,'us')
    objects=[lanes('tick','FIFO service inside one permission period',['Queued traffic','Snapshot read','Constraints','Drive-frame write'],0,period,[0,sense,wcet,deadline,period],0,'Execution progress, µs; blocking and interference charged separately'),obj('memory','pool','FIFO backlog groups and sensor snapshot',dict(slots=4,columns=4,slotLabels=['Group A','Group B','Group C','Snapshot']),1,'memory'),obj('local','chamber','Local permission state',dict(slots=2),2,'compute'),line('completion','Tick completion and proof boundaries, µs',0,period,[0,wcet,deadline,period],3)]
    def case(count,one_channel=False):
        queue_time=n(Q.queue_1ch if one_channel else Q.queue,'us')*count;read=n(Q.own_read_1ch if one_channel else Q.own_read,'us');read_end=queue_time+read;eval_end=read_end+constraint;end=eval_end+write;response=end+bpi
        quotient,remainder=divmod(count,3);groups=[quotient+(i<remainder) for i in range(3)]
        events=[('ready','Prepare the permission snapshot','The sensor snapshot is retained; no competing backlog has yet been enqueued.'),('enqueue','Enqueue competing traffic','The stipulated traffic arrives ahead of the urgent read under FIFO service.'),('serve','Serve the FIFO backlog','The controller completes the already queued traffic before it can serve the snapshot read.'),('read','Read the retained snapshot','The local permission path receives a copy. Reading preserves the sensor-owned snapshot.'),('evaluate','Evaluate the constraints','Constraint evaluation consumes the local snapshot copy and produces an admission decision.'),('write','Write the drive frame','The admitted decision is staged for the drive bus; projected response also includes the separate blocking and interference budget.')]
        steps=[]
        for i,(id,title,ex) in enumerate(events):
            pool=[dict(value=f'{g} backlogs',owner='Competing traffic',status='ready') if i==1 else None for g in groups]+[dict(value='Sensor epoch S',owner='Sensor snapshot',status='retained')]
            segs=[seg('backlog',0,0,queue_time,f'{count} FIFO backlogs','done' if i>=2 else 'waiting','communication'),seg('readTask',1,queue_time,read_end,'Snapshot copy','done' if i>=3 else 'waiting','memory'),seg('evaluateTask',2,read_end,eval_end,'Constraint evaluation','done' if i>=4 else 'waiting'),seg('writeTask',3,eval_end,end,'Frame write','done' if i>=5 else 'waiting','output')]
            cursor=[0,0,queue_time,read_end,eval_end,end][i]
            states={'tick':lane_state(segs,cursor),'memory':dict(slots=pool,status='Groups aggregate repeated backlogs; slots do not model DRAM capacity'),'local':dict(values=['Epoch S' if i>=3 else None,'Admit' if i>=4 else None],status='Local copy and computed decision'),'completion':ls([lm('execution',end,'Execution endpoint'),lm('response',response,'Response including B + I'),lm('declared',wcet,'Declared execution bound','reference'),lm('deadline',deadline,'Response deadline','reference')])}
            result=f'Projected execution {end:.2f} µs; response {response:.2f} µs. '+('The declared execution bound is exceeded.' if end>wcet else 'The declared execution bound holds in this model.')+(' The response misses its deadline.' if response>deadline else ' The response fits its deadline.')
            s=step(id,title,ex,states,[metric('backlogs','Queued backlog count',count if i==1 else 0,'backlogs'),metric('readTime','Loaded snapshot read',read_end,'µs'),metric('execution','Projected execution',end,'µs'),metric('response','Projected response',response,'µs')],result,subjects=['tick','memory','local'],first=i==0)
            if i>=3:
                s['snapshot']['tokens']=[dict(id='snapshotCopy',label='S',value='Sensor epoch S',role='input',owner='Permission path',at=dict(object='local',index=0),state='copy',sourceAt=dict(object='memory',index=3))]
            if i==3:s['transition'].update(kind='copy',subjects=['memory'],**{'from':dict(object='memory',index=3),'to':dict(object='local',index=0)})
            if i==5:states['local']['values'][1]='Frame: Admit'
            steps.append(s)
        return steps
    base_count=Q.backlogs_wcet+1
    return lesson('fig-13-loaded-tick','Queued traffic can break the declared bound','Can a tick finish inside its period while violating its timing premise?', 'The chapter’s stipulated FIFO model at optimistic peak bandwidth. Base case uses the first backlog count beyond the declared WCET threshold. Aggregated pool slots denote groups of whole backlogs, not physical DRAM allocations. This is not measured worst-case service.', 'schedule',objects,case(base_count),['The sensor snapshot remains resident after the read.','Queued traffic is served before the safety read; no priority bypass is assumed.','Declared execution bound, response deadline, and scheduling period are different tests.','Response equals execution plus the chapter’s separate blocking and interference allocation.'],variants=[dict(id='sense-fit',label='Sense allowance boundary',scope='Same interleaved FIFO model at the maximum backlog count within the sense allowance.',steps=case(Q.backlogs_sense)),dict(id='deadline-miss',label='First response deadline miss',scope='Same interleaved FIFO model, one whole backlog beyond the response deadline threshold.',steps=case(Q.backlogs_deadline+1)),dict(id='one-channel',label='Traffic confined to one channel',scope='Same transfers and row-conflict assumptions, served on one channel instead of all interleaved channels; first count beyond its declared WCET threshold.',steps=case(Q.backlogs_wcet_1ch+1,True))])

def thermal():
    recover=n(H.t_recov,'s');trip=H.t_start;clear=H.t_clear;target=H.t_target;tau=n(H.tau_th,'s');period=n(M.tick,'us');deadline=n(M.deadline,'us')
    nominal=n(H.t_exec,'us');slow=n(H.t_exec_throt,'us');bpi=n(H.b_plus_i,'us')
    objects=[plot('temperature','Thermal state retains the prior load',axis(0,4,'Time after removing proposal load, s',[0,1,2,3,4]),axis(50,100,'Junction temperature, °C',[53,85,95]),0),lanes('tick','Same work under the current clock',['Execution','Blocking + interference'],0,period,[0,250,deadline,600,period],1,'Response progress, µs'),obj('load','pool','Proposal load on the shared package',dict(slots=1,slotLabels=['Proposal inference']),2,'compute'),reg('clock','Governor clock','MHz',3)]
    events=[('loaded','Reach the trip threshold',0,True),('remove','Stop proposal inference',0,False),('cool-one','Cool for one second',1,False),('cool-two','Cool for another second',2,False),('clear','Reach the lower clear threshold',recover,False)]
    steps=[]
    for i,(id,title,t,loaded) in enumerate(events):
        temp=trip if loaded else target+(trip-target)*math.exp(-t/tau);restored=i==len(events)-1;execution=nominal if restored else slow;clock=n(H.f_nom if restored else H.f_throt,'MHz');response=execution+bpi
        points=[(t*q/40,target+(trip-target)*math.exp(-(t*q/40)/tau)) for q in range(41)]
        ss=[series('cooling','Model cooling' if not loaded else 'Initial thermal state',points),series('trip','Trip threshold',[(0,trip),(4,trip)],'reference'),series('clear','Clear threshold',[(0,clear),(4,clear)],'reference')]
        segments=[seg('work',0,0,execution,'Same tick work'),seg('bpi',1,execution,response,'Blocking + interference')]
        states={'temperature':ps(ss,[marker('junction',t,temp,'Current junction')]),'tick':lane_state(segments,response,'Schedule geometry changes only when the governor restores the clock'),'load':dict(slots=[dict(value=f'{H.p_load:g} W package',owner='Proposal workload',status='allocated') if loaded else None],status='Proposal workload present' if loaded else f'Baseline package power {H.p_base:g} W remains'),'clock':rs(clock)}
        ex=['Sustained proposal load has heated the shared package to the governor trip threshold.','Removing proposal load changes power immediately; stored heat and the reduced clock remain.','The package cools toward baseline, but the lower clear threshold has not been reached.','Another second of cooling still leaves the governor throttled.','The lower hysteresis threshold permits restoration of the nominal clock.'][i]
        result=f'At {temp:.2f} °C the clock is {clock:g} MHz and response is {response:.0f} µs; '+('the deadline is met.' if response<=deadline else f'the {deadline:g} µs deadline is still missed.')
        steps.append(step(id,title,ex,states,[metric('elapsed','Cooldown elapsed',t,'s'),metric('temperature','Junction temperature',temp,'°C'),metric('response','Tick response',response,'µs'),metric('slack','Deadline slack',deadline-response,'µs')],result,subjects=['temperature','tick','load','clock'],first=i==0))
    return lesson('fig-13-thermal-hysteresis','The load stops before the timing penalty does','Why does a stopped proposer leave later permission ticks too slow?', 'The chapter’s illustrative shared-package RC and governor model, not a chip measurement. The same tick cycle count runs at nominal or halved clock; blocking and interference stay fixed. Each displayed tick is a representative completion, not a repeated animation.', 'control',objects,steps,['Removing workload does not instantly remove stored thermal energy.','The reduced clock persists until the lower clear threshold is reached.','The same instruction work changes duration only because the declared clock changes.','The scheduling period is longer than the deadline; completion inside a period does not prove timely response.'])

def handshake():
    # Qualitative one-channel protocol; normalized command values are declared toy endpoints.
    start,target=0.2,0.6
    objects=[obj('authority','pool','Exactly one selected proposal source',dict(slots=1,slotLabels=['Arm channel']),0,'control'),obj('endpoints','sequence','Latched blend endpoints',dict(slots=2,slotLabels=['Autonomous endpoint','Checked manual target']),1,'memory'),line('command','Admitted command, normalized units',0,1,[0,0.2,0.4,0.6,1],2),obj('protocol','chamber','Authority state, acknowledgment, manual lease',dict(slots=3),3,'control')]
    records=[('auto','Establish autonomous proposal authority','AUTO','Policy',False,False,start),('request','Request the arm channel','PEND','Policy',False,False,start),('ack','Acknowledge the checked target','PEND','Policy',True,False,start),('commit','Commit on a permission tick','BLEND','Enforcer blend',True,True,start),('blend','Execute the admitted blend midpoint','BLEND','Enforcer blend',True,True,(start+target)/2),('manual','Complete the blend with a live manual lease','MAN','Operator',True,True,target),('lapse','Expire the manual lease','FALLBACK','Validated fallback',True,False,target)]
    def make(timeout=False):
        rr=records[:3]+[('timeout','Expire the pending request','FALLBACK','Validated fallback',True,False,start)] if timeout else records
        steps=[]
        for i,(id,title,state,owner,ack,lease,value) in enumerate(rr):
            latched=state in ('BLEND','MAN') or (state=='FALLBACK' and not timeout)
            ss={'authority':dict(slots=[dict(value=owner,owner=owner,status='ready')],status='Actuator bus writer is always the enforcer; this slot names the admitted source'),'endpoints':dict(values=[start,target] if latched else [None,None],status='Immutable after commit' if latched else 'No committed blend endpoints'),'command':ls([lm('admitted',value,'Admitted command'),lm('target',target,'Checked target','reference')]),'protocol':dict(values=[state,'ACK' if ack else None,'LIVE' if lease else None],status='Lease state controls manual eligibility')}
            ex={'auto':'The policy proposes; the enforcer alone writes admitted commands to the actuator bus.','request':'A scoped request starts the handover timer without transferring proposal authority.','ack':'Acknowledgment checks the state and target. The policy retains its proposal token until commit.','commit':'One local permission-tick commit revokes policy proposals, latches both endpoints, and selects the enforcer-owned blend.','blend':'The enforcer advances a feasible, already admitted blend between immutable endpoints.','manual':'The profile completes and a live manual lease permits operator proposals; the enforcer still screens and writes commands.','lapse':'Lease expiry revokes manual proposals and selects the channel’s validated fallback. Its physical trajectory is outside this protocol example.','timeout':'The pending timer expires before commit. The channel selects validated fallback without latching a blend.'}[id]
            steps.append(step(id,title,ex,ss,[metric('owners','Selected channel source count',sum(s is not None for s in ss['authority']['slots']),'sources'),metric('latched','Committed endpoints',sum(v is not None for v in ss['endpoints']['values']),'values')],f'{state}: {owner} is the only selected source. '+('The fallback command initially preserves the current value; validated hold/stop evolves afterward.' if state=='FALLBACK' else 'The enforcer remains the only actuator bus writer.'),subjects=['authority','endpoints','command','protocol'],first=i==0))
        return steps
    return lesson('fig-14-authority-handshake-fsm','Only commit changes channel authority','Why does acknowledgment leave authority unchanged until commit?', 'One arm channel, qualitative handshake, and normalized toy blend endpoints. The blend midpoint illustrates interpolation only; dynamic qualification and physical fallback execution are outside scope. Bus ownership stays with the enforcer even when the selected proposer changes.', 'ownership',objects,make(),['A channel always has exactly one selected proposal source.','Request and acknowledgment do not grant operator proposal authority.','Only commit latches endpoints and revokes the policy proposal token.','Manual proposals require both completed blend and a live manual lease.','Fallback cannot return directly to autonomous or manual authority without standstill and fresh re-admission.'],checks=[dict(kind='metricEqualsOccupied',metric='owners',object='authority'),dict(kind='metricEqualsNonNull',metric='latched',object='endpoints')],variants=[dict(id='timeout',label='Pending request times out',scope='Same channel and checked endpoints, but the request timer expires before commit. No blend endpoints are latched and operator proposal authority is never granted.',steps=make(True))])

def fault():
    v=n(F.v_aisle,'m/s');a=n(F.a_brake,'m/s^2');j=n(F.j_brake,'m/s^3');lease=n(F.t_lease,'ms');tick=n(F.t_tick,'ms');bus=n(F.t_bus,'ms');bound=n(F.onset_bound,'ms');late=n(F.onset_late,'ms');stop_time=v/a+a/j;oracle=n(F.f1_aisle_stop_bound,'mm')
    ramp=a/j;plateau=stop_time-2*ramp
    def speed(t,onset):
        s=max(0,(t-onset)/1000)
        if s<=ramp:return v-j*s*s/2
        v1=v-j*ramp*ramp/2
        if s<=ramp+plateau:return v1-a*(s-ramp)
        v2=v1-a*plateau;u=s-ramp-plateau
        if s<=stop_time:return max(0,v2-a*u+j*u*u/2)
        return 0
    def distance(t,onset):
        # Exact integral of each piecewise-polynomial segment, from renewal.
        pre=v*min(t,onset);s=max(0,min((t-onset)/1000,stop_time));u=min(s,ramp)
        d=v*u-j*u**3/6
        if s>ramp:
            v1=v-j*ramp*ramp/2;u=min(s-ramp,plateau);d+=v1*u-a*u*u/2
        if s>ramp+plateau:
            v2=v-j*ramp*ramp/2-a*plateau;u=s-ramp-plateau;d+=v2*u-a*u*u/2+j*u**3/6
        return pre+d*1000
    objects=[lanes('events','Witnessed F1 events from last valid renewal',['Lease timer','Stop command','Drive frame','Powered deceleration'],0,130,[0,60,82,112,130],0,'Time since last valid renewal, ms'),plot('speed','Modeled base speed to standstill',axis(0,1200,'Time since last valid renewal, ms',[0,400,800,1200]),axis(0,1.5,'Base speed, m/s',[0,0.5,1,1.5]),1),line('distance','Integrated travel, mm',0,850,[0,400,r(oracle),850],2),reg('onsetMargin','Independent onset margin','ms',3)]
    def case(onset):
        end=onset+stop_time*1000
        times=[0,lease,lease+tick,lease+tick+bus,onset,onset+stop_time*500,end]
        ids=['stall','expiry','tick','bus','onset','braking','rest']
        titles=['Stall the proposer after its last renewal','Expire the chunk lease','Issue the stop on the next tick','Deliver the stop frame','Observe powered deceleration onset','Integrate through the resident stop midpoint','Reach standstill and evaluate both predicates']
        steps=[]
        for i,t in enumerate(times):
            spans=[(0,lease,'Lease valid'),(lease+tick,130,'Stop command'),(lease+tick+bus,130,'Drive frame'),(onset,130,'Powered deceleration')]
            segments=[seg('event'+str(q),q,s,e,label,'done' if (q==0 and i>=1) or (q>0 and i>=q+1) else 'waiting','control') for q,(s,e,label) in enumerate(spans)]
            d=distance(t,onset);points=[(t*q/100,speed(t*q/100,onset)) for q in range(101)]
            states={'events':lane_state(segments,min(t,130),'Witness instruments are requirements, not simulated measurements'),'speed':ps([series('trace','Canonical model trace',points)],[marker('current',t,speed(t,onset),'Current model speed')]),'distance':ls([lm('travel',d,'Integrated nominal travel'),lm('oracle',oracle,'Distance oracle including tracking allowance','reference')]),'onsetMargin':rs(bound-onset if i>=4 else None,'Scored from witnessed onset' if i>=4 else 'Not scored until onset is witnessed',f'{bound:g} ms bound − {onset:g} ms onset' if i>=4 else f'Bound {bound:g} ms; onset not witnessed')}
            ex=['A modeled stall prevents renewal. All event times share the last valid renewal as their reference.','The local lease expires; it does not itself mark physical braking onset.','The next permission tick emits the stop command.','Bus delivery precedes the actuator’s powered deceleration.','A witnessed onset can now be compared with its independent deadline.','The area under the modeled speed curve accumulates travel while braking continues.','Standstill permits the distance verdict. Distance slack cannot erase a negative onset margin.'][i]
            verdict='Unfinished model trace.' if i<6 else ('Both predicates pass in this model.' if onset<=bound and d<=oracle else 'Onset fails while distance passes in this model.' if d<=oracle else 'Distance predicate fails.')
            steps.append(step(ids[i],titles[i],ex,states,[metric('elapsed','Elapsed from renewal',t,'ms'),metric('travel','Integrated travel',d,'mm'),metric('distanceMargin','Distance allowance remaining',oracle-d,'mm'),metric('onsetMargin','Onset margin',bound-onset if i>=4 else 'Not witnessed','ms')],verdict,subjects=['events','speed','distance','onsetMargin'],first=i==0))
        assert abs(distance(end,onset)-(v*onset+n(F.c2_distance,'mm')))<1e-8
        return steps
    return lesson('fig-15-hardware-fault-injection','A stop must satisfy two independent margins','Can a stop pass its travel oracle while failing its onset oracle?', 'Canonical F1 model trace referenced to the last valid renewal. Base case begins braking at the onset bound; late case delays onset by the owning LEGO’s illustrative amount. Piecewise jerk-limited speed is analytically integrated. No qualification measurements or missing witnesses are invented.', 'control',objects,case(bound),['All times and integrated travel use the last valid renewal as the same reference event.','Tracking inset is an oracle allowance, not extra nominal travel.','Onset and distance margins retain their distinct physical units.','A missing required physical witness makes a real trial inconclusive, regardless of this complete model trace.'],variants=[dict(id='late',label='Late powered deceleration',scope='Same model speed and jerk-limited resident stop, with the chapter’s declared late onset. The independent onset and distance limits remain unchanged.',steps=case(late))])

def build():return [nonholonomic(),latch(),capture(),queue(),thermal(),handshake(),fault()]
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--write',action='store_true',help='Explicit authoring mode; default checks without rewriting');parser.add_argument('--json',type=pathlib.Path,default=pathlib.Path(__file__).with_name('vol4-c.json'));args=parser.parse_args()
    expected=build()
    if args.write:args.json.write_text(json.dumps(expected,ensure_ascii=False,indent=2)+'\n')
    actual=json.loads(args.json.read_text());assert actual==expected,'Authored JSON differs from recomputed complete lesson states'
    for lesson_ in expected:
        for name,steps in [('original',lesson_['steps'])]+[(v['id'],v['steps']) for v in lesson_.get('variants',[])]:
            last=steps[-1];print(lesson_['key'],name,last['id'],json.dumps(last['snapshot']['metrics'],ensure_ascii=False))
    print('PASS: complete states, geometry, counts, metrics, and all variant endpoints recomputed from canonical LEGO or declared schematic inputs.')
if __name__=='__main__':main()

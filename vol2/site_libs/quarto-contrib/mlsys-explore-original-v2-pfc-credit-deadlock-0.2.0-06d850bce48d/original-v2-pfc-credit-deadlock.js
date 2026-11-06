(() => {
'use strict';
const NS='http://www.w3.org/2000/svg';
const el=(tag,a={},text)=>{const n=document.createElementNS(NS,tag);Object.entries(a).forEach(([k,v])=>n.setAttribute(k,v));if(text!==undefined)n.textContent=text;return n;};
window.MLSysFigureLessons=window.MLSysFigureLessons||{};
window.MLSysFigureLessons["original-v2-pfc-credit-deadlock"]=async(root,context)=>{
 if(!context.verifiedArtwork||!window.MLSysOriginalPlayer)throw Error('Verified source artwork and shared player required');
 if(context.verifiedArtwork.getAttribute('viewBox')!=="0 -4 680 420")throw Error('Source geometry changed; review required');
 root.classList.add('v2c-lesson');
 const art=el('svg',{viewBox:"0 0 680 430",role:'img','aria-label':"What must change before a deadlocked queue can forward?"});
 const base=el('g'),moving=el('g',{'aria-hidden':'true'});art.append(base,moving);
 const text=(x,y,t,a={})=>el('text',{x,y,'font-size':14,fill:'#14294e',...a},t);
 const box=(x,y,w,h,label,fill='#edf2fa',stroke='#45699b')=>{const g=el('g');g.append(el('rect',{x,y,width:w,height:h,rx:2,fill,stroke}),text(x+w/2,y+h/2+5,label,{'text-anchor':'middle','font-weight':600}));return g;};
 const line=(pts,color='#8b9bb0',dash='')=>{const g=el('g'),a=pts.at(-2),b=pts.at(-1),theta=Math.atan2(b[1]-a[1],b[0]-a[0]),tip=[b,[b[0]-8*Math.cos(theta)+4*Math.sin(theta),b[1]-8*Math.sin(theta)-4*Math.cos(theta)],[b[0]-8*Math.cos(theta)-4*Math.sin(theta),b[1]-8*Math.sin(theta)+4*Math.cos(theta)]];g.append(el('polyline',{points:pts.map(p=>p.join(',')).join(' '),fill:'none',stroke:color,'stroke-width':2,'stroke-dasharray':dash}),el('polygon',{points:tip.map(p=>p.join(',')).join(' '),fill:color}));return g;};
 const token=(x,y,label,color='#edf2fa',attrs={})=>{const g=el('g',{'data-object':label,...attrs}),w=Math.max(28,label.length*7+10);g.append(el('rect',{x:x-w/2,y:y-12,width:w,height:24,rx:2,fill:color,stroke:'#45699b'}),text(x,y+5,label,{'text-anchor':'middle','font-size':12,'font-weight':700}));return g;};
 const status=document.createElement('p');status.className='v2c-status';status.setAttribute('aria-live','polite');
 const note=document.createElement('p');note.className='v2c-reading';note.textContent="Queue contents and the free slot are authoritative at every stopped state. Solid gray dependencies point clockwise; red PAUSE messages point counterclockwise. Recovery here is a declared toy, not a universal PFC implementation.";
 note.textContent+=' On narrow screens, the view follows the active transfer; pause to swipe across the full diagram.';
 let mounted=false,stageNode=null;
 function camera(x){if(!stageNode)return;const scale=art.getBoundingClientRect().width/art.viewBox.baseVal.width;stageNode.scrollLeft=Math.max(0,x*scale-stageNode.clientWidth/2);}
 const centers=[[340,62],[560,211],[340,360],[120,211]],queues=[Array.from({length:4},(_,q)=>Array.from({length:4},(_,i)=>'ABCD'[q]+i))];
for(let i=1;i<8;i++){let q=queues.at(-1).map(a=>[...a]);if(i===3)q[0].shift();if(i>=4){const sender=[3,2,1,0][i-4],receiver=(sender+1)%4;q[receiver].push(q[sender].shift());}queues.push(q);}
const pausePaths=[[[300,125],[240,190]],[[440,190],[390,125]],[[390,303],[440,232]],[[240,232],[300,303]]];
const edge=(q,reverse=false)=>{const a=centers[q],b=centers[(q+1)%4],m=[(a[0]+b[0])/2+(q<2?40:-40),(a[1]+b[1])/2+(q===0||q===3?-25:25)];return reverse?[b,m,a]:[a,m,b];};
 const controller=window.MLSysOriginalPlayer.create({root,outer:context.outer,title:"What must change before a deadlocked queue can forward?",scope:"The original four-switch cycle, with four declared toy packet slots per queue. For this toy only, full queues pause their predecessor and any free slot allows resume. A watchdog drops one packet; real thresholds and recovery policies vary.",steps:[{"label": "All buffers full", "description": "Each queue holds four packets. Its head packet needs a slot in the next queue clockwise, but no next queue has space."}, {"label": "PAUSE travels upstream", "description": "Each full queue sends PAUSE counterclockwise to its predecessor. Packet forwarding dependencies still point clockwise."}, {"label": "Deadlock", "description": "No packet can advance: every head waits for a full downstream queue, and every sender is paused. Waiting alone cannot make space."}, {"label": "Watchdog drops A0", "description": "An external watchdog drops A0 from queue 1. This is packet loss, not successful delivery. The new free slot allows queue 4 to resume in this declared toy."}, {"label": "Queue 4 forwards D0", "description": "D0 advances clockwise from queue 4 into the free slot in queue 1. Queue 4 now has the free slot."}, {"label": "Queue 3 forwards C0", "description": "C0 advances clockwise into queue 4. The free slot moves upstream to queue 3."}, {"label": "Queue 2 forwards B0", "description": "B0 advances clockwise into queue 3. Queue 2 can now accept a packet."}, {"label": "Queue 1 forwards A1", "description": "A1 advances clockwise into queue 2. The cycle has made progress because recovery created space; the discarded A0 still requires transport recovery."}].map((step,i)=>({label:step.label,get description(){return Number(root.dataset.modelState)===i?step.description:"On arrival: "+step.description;}})),
 render(index,{stage,details}){
  if(!mounted){stageNode=stage;stage.classList.add('v2c-stage');stage.tabIndex=0;stage.setAttribute('aria-label','Mechanism diagram; swipe horizontally on a narrow screen or use Expand');stage.append(art);details.append(status,note);mounted=true;}
  base.replaceChildren();moving.replaceChildren();root.dataset.modelState=String(index);
  const q=queues[index];root.dataset.queues=JSON.stringify(q);root.dataset.dropped=index>=3?'A0':'';root.dataset.packetCount=String(q.reduce((n,a)=>n+a.length,0));root.dataset.deadlocked=String(index<=2);
for(let s=0;s<4;s++){const p=edge(s);base.append(line(p,'#64748b'));const m=p[1];base.append(text(m[0],m[1]-9,`Q${s+1} → Q${(s+1)%4+1}`,{'text-anchor':'middle','font-size':12}));}
for(let s=0;s<4;s++){const[x,y]=centers[s],full=q[s].length===4;base.append(box(x-90,y-37,180,80,'','#fff3cd','#d97706'),text(x,y-18,`Switch ${s+1} / Q${s+1}`,{'text-anchor':'middle','font-weight':700}));for(let slot=0;slot<4;slot++){const label=q[s][slot]||'free';base.append(el('rect',{x:x-73+slot*36,y:y-5,width:34,height:25,fill:label==='free'?'#dcfce7':'#fee2e2',stroke:label==='free'?'#16a34a':'#ef4444'}),text(x-56+slot*36,y+12,label,{'text-anchor':'middle','font-size':12,'data-packet':label}));}base.append(text(x,y+36,full?'full · pauses upstream':'free slot · upstream may resume',{'text-anchor':'middle','font-size':11,fill:full?'#b91c1c':'#15803d'}));
if(index>=1&&full){const prev=(s+3)%4,inner=pausePaths[s];base.append(line(inner,'#dc2626','5 4'));const mx=(inner[0][0]+inner[1][0])/2,my=(inner[0][1]+inner[1][1])/2;base.append(text(mx,my+(s<2?-24:28),`PAUSE → Q${prev+1}`,{'text-anchor':'middle','font-size':11,fill:'#b91c1c','data-pause-from':s+1,'data-pause-to':prev+1}));}}
base.append(text(340,205,index<=2?'No legal forward':`Watchdog dropped A0`,{'text-anchor':'middle','font-weight':700,fill:'#b91c1c'}),text(340,228,index<=2?'every next queue is full':`Free slot: Q${q.findIndex(a=>a.length<4)+1}`,{'text-anchor':'middle','font-size':12}));
status.textContent=index<=2?'Clockwise: packets wait for the next queue. Counterclockwise: PAUSE blocks the previous sender. All sixteen packets remain queued.':`A0 dropped, not delivered. ${q.reduce((n,a)=>n+a.length,0)} packets remain queued. Free slot is now in Q${q.findIndex(a=>a.length<4)+1}; progress costs packet loss and later transport recovery.`;
 camera(index<3?340:centers[[0,3,2,1,0][index-3]][0]);
 },
 transition({from,to,onFrame}){
  if(to!==from+1)return;
  const motions=to===1?centers.map((a,s)=>({label:'PAUSE',color:'#fee2e2',path:pausePaths[s]})):to>=4?[{label:['D0','C0','B0','A1'][to-4],path:edge([3,2,1,0][to-4])}]:to===3?[{label:'A0 ×',color:'#fee2e2',path:[[340,62],[340,140]]}]:[];
  if(to>=3){const id=to===3?'A0':['D0','C0','B0','A1'][to-4],label=base.querySelector(`[data-packet="${id}"]`);if(label){label.textContent=to===3?'pending':'in flight';label.setAttribute('font-size','8');label.dataset.inflight=id;label.removeAttribute('data-packet');const rect=label.previousElementSibling;rect.setAttribute('fill','#fff');rect.setAttribute('stroke-dasharray','3 2');}}
  status.textContent="Transition in progress; destination state is pending. "+status.textContent;
  if(motions.length===16)base.querySelectorAll("[data-live-token]").forEach(n=>n.setAttribute("visibility","hidden"));
  const packets=motions.map(m=>{const g=token(0,0,m.label,m.color||'#fff3cd',{'data-motion-object':'true'});moving.append(g);const lengths=m.path.slice(1).map((p,i)=>Math.hypot(p[0]-m.path[i][0],p[1]-m.path[i][1]));return {...m,g,lengths,total:lengths.reduce((a,b)=>a+b,0)};});
  onFrame(p=>{root.dataset.animationProgress=String(p);packets.forEach(m=>{let d=p*m.total,i=0;while(i<m.lengths.length-1&&d>m.lengths[i])d-=m.lengths[i++];const a=m.path[i],b=m.path[i+1],f=m.lengths[i]?d/m.lengths[i]:1;const px=a[0]+(b[0]-a[0])*f;m.g.setAttribute('transform',`translate(${px},${a[1]+(b[1]-a[1])*f})`);if(m===packets[0])camera(px);});});
 }});
 return controller;
};
})();

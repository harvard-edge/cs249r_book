/* Original-artwork mechanism. Shared player owns the pausable clock. */
(() => {'use strict';
window.MLSysFigureLessons=window.MLSysFigureLessons||{};
window.MLSysFigureLessons["original-v4-12-capture-to-onset"]=async(root,context)=>{
 if(!context.verifiedArtwork||!window.MLSysOriginalPlayer)throw Error('Verified original artwork and shared player required');
 const svg=context.verifiedArtwork.cloneNode(true),ns='http://www.w3.org/2000/svg';
 const originalView=svg.getAttribute('viewBox');if(originalView!=="0 0 3913 2974")throw Error('Original geometry changed; retain Diagram for review');
 root.classList.add("mlsys-original-v4-12-capture-to-onset");
 const canvas=document.createElement('div');canvas.className='v4b-canvas';canvas.tabIndex=0;canvas.setAttribute('aria-label','Original artwork. Scroll to inspect; Expand opens a larger view.');canvas.append(svg);
 svg.setAttribute('role','img');svg.setAttribute('aria-label',"Which clock pays for the path to brake onset?");
 const overlay=document.createElementNS(ns,'g');svg.append(overlay);
 const add=(tag,attrs={},text)=>{const el=document.createElementNS(ns,tag);for(const[k,v]of Object.entries(attrs))el.setAttribute(k,String(v));if(text!=null)el.textContent=text;overlay.append(el);return el;};
 const readout=document.createElement('p');readout.className='v4b-readout';readout.setAttribute('aria-live','off');
 const equation=document.createElement('p');equation.className='v4b-equation';
 const note=document.createElement('p');note.className='v4b-note';note.textContent='Scroll the original artwork to inspect labels, or use Expand. Playback duration is illustrative.';
 const width=+originalView.split(' ')[2],height=+originalView.split(' ')[3];
 function follow(x,y){const scale=svg.getBoundingClientRect().width/width;canvas.scrollLeft=Math.max(0,x*scale-canvas.clientWidth/2);canvas.scrollTop=Math.max(0,y*scale-canvas.clientHeight/2);}
 function mount(stage,details){if(!canvas.isConnected)stage.append(physicalScene,canvas,note);if(!readout.isConnected)details.append(readout,equation);}
 function dot(x,y,color='#007A87',r=5){return add('circle',{cx:x,cy:y,r,fill:color,stroke:'#fff','stroke-width':r/3,'data-motion-object':'true'});}
 function route(token,points,p){const lens=points.slice(1).map((b,i)=>Math.hypot(b[0]-points[i][0],b[1]-points[i][1]));let rem=p*lens.reduce((a,b)=>a+b,0),j=0;while(j<lens.length-1&&rem>lens[j]){rem-=lens[j];j++;}const q=lens[j]?rem/lens[j]:1,a=points[j],b=points[j+1];token.setAttribute('cx',a[0]+(b[0]-a[0])*q);token.setAttribute('cy',a[1]+(b[1]-a[1])*q);}
 const D={"v": 1.3, "dispatch": 51.6, "allocation": 63.31072, "lease": 60.0, "tick_bus": 2.0, "onset_delay": 20.0, "allowance": 100.77500000000003, "spare": 131.00750000000005, "admission": 114.91072, "expiry": 174.91072, "tick_end": 176.91072, "onset": 196.91072, "latest_admission": 152.37500000000003, "latest_onset": 234.37500000000003, "reserve_ms": 37.46428000000003, "reserve_mm": 48.70356400000004};

 const X=t=>968+(3665-968)*t/250;
 const cursor=add('line',{x1:X(0),x2:X(0),y1:500,y2:2000,stroke:'#007A87','stroke-width':9,'stroke-dasharray':'20 14','data-motion-object':'true'});
 const travel=add('line',{x1:X(0),x2:X(0),y1:2300,y2:2300,stroke:'#007A87','stroke-width':22,'data-motion-object':'true'});
 const base=add('g',{'data-motion-object':'true'});const chassis=add('rect',{x:-55,y:-45,width:110,height:55,fill:'#fff',stroke:'#007A87','stroke-width':8});base.append(chassis);for(const x of [-33,33]){const wheel=add('circle',{cx:x,cy:15,r:15,fill:'#007A87'});base.append(wheel);}
 const spans=[[0,D.dispatch,1035,'#007A87'],[D.dispatch,D.admission,1330,'#B87333'],[D.admission,D.expiry,1630,'#1A4D3E'],[D.expiry,D.tick_end,1630,'#1A4D3E'],[D.tick_end,D.onset,1920,'#A51C30']].map(([a,b,y,c])=>({a,b,line:add('line',{x1:X(a),x2:X(a),y1:y,y2:y,stroke:c,'stroke-width':14,'data-motion-object':'true'})}));

 const physicalScene=document.createElementNS(ns,'svg');physicalScene.setAttribute('viewBox','0 0 500 100');physicalScene.classList.add('v4b-physical');physicalScene.setAttribute('role','img');physicalScene.setAttribute('aria-label','Same pre-onset base travel as original strip; icon size schematic');
 const pg=add('g');physicalScene.append(pg);const miniAdd=(tag,attrs,text)=>{const e=add(tag,attrs,text);pg.append(e);return e;};
 miniAdd('line',{x1:35,x2:435,y1:58,y2:58,stroke:'#718096','stroke-width':2});
 for(const d of [0,100,200,300]){const x=35+d*400/300;miniAdd('line',{x1:x,x2:x,y1:54,y2:63,stroke:'#718096','stroke-width':2});miniAdd('text',{x,y:80,'text-anchor':'middle','font-size':14,fill:'currentColor'},String(d));}
 const miniBase=miniAdd('g',{'data-motion-object':'true'});const miniRect=miniAdd('rect',{x:-22,y:27,width:44,height:23,fill:'#E3F3F5',stroke:'#007A87','stroke-width':3});miniBase.append(miniRect);for(const x of [-13,13]){const wheel=miniAdd('circle',{cx:x,cy:52,r:6,fill:'#007A87'});miniBase.append(wheel);}
 miniAdd('text',{x:250,y:98,'text-anchor':'middle','font-size':13,fill:'currentColor'},'Pre-onset travel (mm) · base size schematic');
 const times=[0,D.dispatch,D.admission,D.expiry,D.tick_end,D.onset];
 const steps=[
 ['Date the observation at capture','Time zero is the exposure midpoint, when the observation was physically true. Exposure/readout are not permission.'],
 ['Age the observation until dispatch','Follow the capture clock through perception. At dispatch its age is 51.6 ms. The lease has not started. Travel is a constant-speed budget conversion, not a new sensor measurement.'],
 ['Charge dispatch-to-admission separately','Follow the hatched allocation until admission. This interval belongs to neither observation age-to-dispatch nor the lease. It is a placement budget that loaded measurement must still qualify.'],
 ['Run the lease from admission','At admission the lease clock begins; it expires 60 ms later. The capture-relative clock keeps its original origin. A fresh lease does not make the captured observation new.'],
 ['Charge one tick and one bus cycle','After lease expiry, one tick and one bus cycle carry the fallback response toward the drive. Both durations are charged exactly once.'],
 ['Reach physical brake onset','After drive-onset delay, braking begins at the endpoint. The stop that follows continues beyond the original plot’s right edge. Compare the actual allocated onset with the dashed latest-onset bound; this timeline does not prove loaded qualification.']
 ].map(([label,description],i)=>({label,get description(){return root.dataset.lessonCommittedIndex===String(i)?description:'Next endpoint, not yet completed: '+description;}}));
 function frame(t){const x=X(t),distance=D.v*t,leaseAge=Math.max(0,t-D.admission);
  cursor.setAttribute('x1',x);cursor.setAttribute('x2',x);travel.setAttribute('x2',x);spans.forEach(s=>s.line.setAttribute('x2',X(Math.min(s.b,Math.max(s.a,t)))));
  base.setAttribute('transform',`translate(${x} 2290)`);miniBase.setAttribute('transform',`translate(${35+distance*400/300} 0)`);physicalScene.setAttribute('aria-label',`Same pre-onset travel ${distance.toFixed(2)} mm; base size schematic`);root.dataset.baseTravelMm=String(distance);root.dataset.time=String(t);root.dataset.travel=String(distance);root.dataset.leaseStarted=String(t>=D.admission);root.dataset.leaseExpired=String(t>=D.expiry);
  readout.textContent=`Capture clock ${t.toFixed(2)} ms · pre-onset travel reservation ${distance.toFixed(2)} mm · ${t<D.admission?'lease not started':t<D.expiry?`lease age ${leaseAge.toFixed(2)} ms of ${D.lease.toFixed(0)} ms`:'lease expired; fallback response'}`;
  equation.textContent=`d = v t = ${D.v.toFixed(1)} m/s × ${t.toFixed(2)} ms = ${distance.toFixed(2)} mm. Latest admission ${D.latest_admission.toFixed(2)} ms; latest onset ${D.latest_onset.toFixed(2)} ms. Allocation leaves ${D.reserve_ms.toFixed(2)} ms = ${D.reserve_mm.toFixed(2)} mm reserve.`;
 }
 function render(i,{stage,details}){root.dataset.lessonCommittedIndex=String(i);mount(stage,details);frame(times[i]);follow(X(times[i]),[660,960,1250,1550,1550,1845][i]);}
 function transition({from,to,onFrame}){if(to!==from+1)return;onFrame(p=>{frame(times[from]+(times[to]-times[from])*p);root.dataset.animationProgress=String(p);});}

 return window.MLSysOriginalPlayer.create({root,outer:context.outer,title:"Which clock pays for the path to brake onset?",scope:"Original timing/travel strip, exact canonical interval endpoints at the chosen aisle speed. Dispatch allocation is unqualified; constant-speed travel maps the pre-onset budget only. Base position follows this travel row; icon size is schematic.",steps,render,transition});
};})();

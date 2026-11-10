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
 // Keep the endpoint in view when Expand or rotation changes the artwork scale.
 // Scrolling alone does not trigger this observer, so manual inspection persists.
 let focus=null,resizeFrame=0,disposed=false;
 function recenter(){if(!focus||disposed)return;const scale=svg.getBoundingClientRect().width/width;canvas.scrollLeft=Math.max(0,focus[0]*scale-canvas.clientWidth/2);canvas.scrollTop=Math.max(0,focus[1]*scale-canvas.clientHeight/2);}
 function follow(x,y){focus=[x,y];recenter();}
 const cameraResize=typeof ResizeObserver==='function'?new ResizeObserver(()=>{if(!disposed&&!resizeFrame)resizeFrame=requestAnimationFrame(()=>{resizeFrame=0;recenter();});}):null;
 cameraResize?.observe(canvas);cameraResize?.observe(svg);
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
 ['Date the observation at capture','At capture origin t = 0.00 ms (0.00 mm pre-onset travel at v = 1.3 m/s), the top sensor row marks the exposure midpoint when the world observation was physically true; readout and transport have not begun and no permission lease exists yet.'],
 ['Age the observation until dispatch','Across the teal perception row from 0.00 to 51.6 ms, exposure, readout, and inference age the observation by 51.6 ms while the base advances d = 1.3 × 51.6 = 67.08 mm before any candidate proposal is dispatched.'],
 ['Charge dispatch-to-admission separately','From 51.6 ms to 114.91 ms, the hatched bronze dispatch-to-admission allocation consumes 63.31 ms (63.3 ms rounded, 82.30 mm of travel to reach 149.38 mm cumulative), leaving 37.46 ms (48.70 mm) of unused reserve before the 152.38 ms latest-admission bound.'],
 ['Run the lease from admission','At t = 114.91 ms, the permission core admits the command and starts the 60 ms green lease window; when that 60 ms lease expires at 174.91 ms (227.38 mm cumulative travel), the underlying sensor evidence is already 174.91 ms old.'],
 ['Charge one tick and one bus cycle','Between 174.91 ms and 176.91 ms, lease expiry triggers the local fallback path across 1 ms of permission-tick quantization plus 1 ms of fieldbus delivery (2.00 ms total, adding 2.60 mm to reach 229.98 mm cumulative travel).'],
 ['Reach physical brake onset','From 176.91 ms to 196.91 ms (196.9 ms), the red 20.00 ms actuator onset delay adds 26.00 mm so physical deceleration begins at 196.91 ms and 255.98 mm pre-onset travel—37.46 ms (48.70 mm) inside the 234.38 ms latest-onset bound.']
 ].map(([label,description],i)=>({label,get description(){return root.dataset.lessonCommittedIndex===String(i)?description:'Next endpoint, not yet completed: '+description;}}));
 function frame(t){const x=X(t),distance=D.v*t,leaseAge=Math.max(0,t-D.admission);
  cursor.setAttribute('x1',x);cursor.setAttribute('x2',x);travel.setAttribute('x2',x);spans.forEach(s=>s.line.setAttribute('x2',X(Math.min(s.b,Math.max(s.a,t)))));
  base.setAttribute('transform',`translate(${x} 2290)`);miniBase.setAttribute('transform',`translate(${35+distance*400/300} 0)`);physicalScene.setAttribute('aria-label',`Same pre-onset travel ${distance.toFixed(2)} mm; base size schematic`);root.dataset.baseTravelMm=String(distance);root.dataset.time=String(t);root.dataset.travel=String(distance);root.dataset.leaseStarted=String(t>=D.admission);root.dataset.leaseExpired=String(t>=D.expiry);
  readout.textContent=`Capture clock ${t.toFixed(2)} ms · pre-onset travel reservation ${distance.toFixed(2)} mm · ${t<D.admission?'lease not started':t<D.expiry?`lease age ${leaseAge.toFixed(2)} ms of ${D.lease.toFixed(0)} ms`:'lease expired; fallback response'}`;
  equation.textContent=`d = v t = ${D.v.toFixed(1)} m/s × ${t.toFixed(2)} ms = ${distance.toFixed(2)} mm. Latest admission ${D.latest_admission.toFixed(2)} ms; latest onset ${D.latest_onset.toFixed(2)} ms. Allocation leaves ${D.reserve_ms.toFixed(2)} ms = ${D.reserve_mm.toFixed(2)} mm reserve.`;
 }
 function render(i,{stage,details}){root.dataset.lessonCommittedIndex=String(i);mount(stage,details);frame(times[i]);follow(X(times[i]),[660,960,1250,1550,1550,1845][i]);}
 function transition({from,to,onFrame}){if(to!==from+1)return;onFrame(p=>{frame(times[from]+(times[to]-times[from])*p);root.dataset.animationProgress=String(p);});}

 const player=window.MLSysOriginalPlayer.create({root,outer:context.outer,layout:'compact',title:"Which clock pays for the path to brake onset?",intro:"Follow the 1.3 m/s permission timeline from capture through the 51.6 ms observation age, 63.3 ms dispatch allocation, and 60 ms lease to physical brake onset at 196.9 ms.",scope:"Aisle base pre-onset timing and travel budget at constant speed v = 1.3 m/s (1.3 mm/ms): observation age at dispatch 51.60 ms (67.08 mm), hatched dispatch-to-admission allocation 63.31 ms (82.30 mm, admission at 114.91 ms vs. 152.38 ms latest admission, leaving 37.46 ms = 48.70 mm reserve out of the 100.78 ms = 131.01 mm spare allowance), permission lease 60.00 ms (78.00 mm, expiry at 174.91 ms), 1 ms tick + 1 ms bus cycle (2.00 ms = 2.60 mm to 176.91 ms), and 20.00 ms brake-onset delay (26.00 mm) reaching physical brake onset at 196.91 ms (255.98 mm vs. 234.38 ms latest onset).",steps,render,transition});
 const destroy=player.destroy;player.destroy=()=>{disposed=true;cameraResize?.disconnect();cancelAnimationFrame(resizeFrame);destroy();};return player;
};})();

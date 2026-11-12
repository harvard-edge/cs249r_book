/* Original-artwork mechanism. Shared player owns the pausable clock. */
(() => {'use strict';
window.MLSysFigureLessons=window.MLSysFigureLessons||{};
window.MLSysFigureLessons["original-v4-10-latch-force-budget"]=async(root,context)=>{
 if(!context.verifiedArtwork||!window.MLSysOriginalPlayer)throw Error('Verified original artwork and shared player required');
 const svg=context.verifiedArtwork.cloneNode(true),ns='http://www.w3.org/2000/svg';
 const originalView=svg.getAttribute('viewBox');if(originalView!=="0 0 3660 2100")throw Error('Original geometry changed; retain Diagram for review');
 root.classList.add("mlsys-original-v4-10-latch-force-budget");
 const canvas=document.createElement('div');canvas.className='v4b-canvas';canvas.tabIndex=0;canvas.setAttribute('aria-label','Original artwork. Scroll to inspect; Expand opens a larger view.');canvas.append(svg);
 svg.setAttribute('role','img');svg.setAttribute('aria-label',"Why size the stop from force at brake onset?");
 const overlay=document.createElementNS(ns,'g');svg.append(overlay);
 const add=(tag,attrs={},text)=>{const el=document.createElementNS(ns,tag);for(const[k,v]of Object.entries(attrs))el.setAttribute(k,String(v));if(text!=null)el.textContent=text;overlay.append(el);return el;};
 const readout=document.createElement('p');readout.className='v4b-readout';readout.setAttribute('aria-live','off');
 const equation=document.createElement('p');equation.className='v4b-equation';
 const note=document.createElement('p');note.className='v4b-note';note.textContent='Scroll the original artwork to inspect labels, or use Expand. Playback duration is illustrative.';
 const width=+originalView.split(' ')[2],height=+originalView.split(' ')[3];
 // Keep the endpoint in view when Expand or rotation changes the artwork scale.
 // Scrolling alone does not trigger this observer, so manual inspection persists.
 let focus=null,resizeFrame=0,disposed=false;
 function recenter(){if(!focus||disposed)return;if(canvas.clientWidth>=520){canvas.scrollLeft=0;canvas.scrollTop=0;return;}const scale=svg.getBoundingClientRect().width/width;canvas.scrollLeft=Math.max(0,focus[0]*scale-canvas.clientWidth/2);canvas.scrollTop=Math.max(0,focus[1]*scale-canvas.clientHeight/2);}
 function follow(x,y){focus=[x,y];recenter();}
 const cameraResize=typeof ResizeObserver==='function'?new ResizeObserver(()=>{if(!disposed&&!resizeFrame)resizeFrame=requestAnimationFrame(()=>{resizeFrame=0;recenter();});}):null;
 cameraResize?.observe(canvas);cameraResize?.observe(svg);
 function mount(stage,details){if(!canvas.isConnected)stage.append(physicalScene,canvas,note);if(!readout.isConnected)details.append(readout,equation);}
 function dot(x,y,color='#007A87',r=5){return add('circle',{cx:x,cy:y,r,fill:color,stroke:'#fff','stroke-width':r/3,'data-motion-object':'true'});}
 function route(token,points,p){const lens=points.slice(1).map((b,i)=>Math.hypot(b[0]-points[i][0],b[1]-points[i][1]));let rem=p*lens.reduce((a,b)=>a+b,0),j=0;while(j<lens.length-1&&rem>lens[j]){rem-=lens[j];j++;}const q=lens[j]?rem/lens[j]:1,a=points[j],b=points[j+1];token.setAttribute('cx',a[0]+(b[0]-a[0])*q);token.setAttribute('cy',a[1]+(b[1]-a[1])*q);}
 const D={"k": 400000.0, "v": 0.03, "trip": 15.0, "limit": 100.0, "response": 2.0, "onset_force": 39.0, "a": 2.9508196721311477, "stop_ms": 10.166666666666666, "trip_ms": 1.25, "onset_ms": 3.25, "end_ms": 13.416666666666666, "unbraked_limit_ms": 8.333333333333334};

 const X=t=>354+(3515-354)*t/D.end_ms,Y=f=>1798-(1798-596)*f/100;
 const actual=dot(X(0),Y(0),'#1A4D3E',25),unbraked=dot(X(0),Y(0),'#A51C30',23);
 const trace=add('path',{d:'',fill:'none',stroke:'#1A4D3E','stroke-width':15,opacity:.55,'data-motion-object':'true'});
 const plate=add('line',{x1:3250,x2:3250,y1:1200,y2:1450,stroke:'#4A5568','stroke-width':12});
 const arm=add('rect',{x:2250,y:1270,width:180,height:95,rx:0,fill:'#E8F0EC',stroke:'#1A4D3E','stroke-width':8,'data-motion-object':'true'});
 const spring=add('path',{d:'',fill:'none',stroke:'#B87333','stroke-width':9,'data-motion-object':'true'});
 add('text',{x:2700,y:1445,'text-anchor':'middle','font-size':56,fill:'#4A5568'},'spring compression (amplified)');

 const physicalScene=document.createElementNS(ns,'svg');physicalScene.setAttribute('viewBox','0 0 500 100');physicalScene.classList.add('v4b-physical');physicalScene.setAttribute('role','img');physicalScene.setAttribute('aria-label','Same arm and spring state as the force plot; compression amplified');
 const pg=add('g');physicalScene.append(pg);
 const miniAdd=(tag,attrs,text)=>{const e=add(tag,attrs,text);pg.append(e);return e;};
 miniAdd('line',{x1:460,x2:460,y1:16,y2:72,stroke:'#718096','stroke-width':4});
 const miniArm=miniAdd('rect',{x:10,y:28,width:65,height:32,fill:'#E8F0EC',stroke:'#1A4D3E','stroke-width':3,'data-motion-object':'true'});
 const miniSpring=miniAdd('path',{fill:'none',stroke:'#B87333','stroke-width':3,'data-motion-object':'true'});
 const miniLabel=miniAdd('text',{x:250,y:92,'font-size':17,'text-anchor':'middle',fill:'currentColor'},'Same arm/plate state · compression amplified');
 const times=[0,D.trip_ms,D.onset_ms,6,D.unbraked_limit_ms,D.end_ms];
 const steps=[
 ['First contact: compression starts','At t = 0.00 ms, the rigid arm touches the latch spring at guarded approach speed v_0 = 0.030 m/s with 0.0 N contact force and 0.0000 mm compression, starting a linear force rise at k v_0 = 12 N/ms.'],
 ['Cross the contact tripwire','At t = 1.25 ms, spring compression reaches 0.0375 mm and contact force crosses the 15 N tripwire; detection triggers the stop command, but the arm still moves at 0.030 m/s while waiting for the 2 ms response window.'],
 ['Wait through response to brake onset','Across the 2 ms detection, command, and torque-rise delay (from 1.25 ms to 3.25 ms), unbraked motion adds k v_0 Δt_resp = 24 N so physical braking begins at 39 N (0.0975 mm compression), leaving 100 - 39 = 61 N of stopping budget.'],
 ['Decelerate while the spring keeps compressing','At t = 6.00 ms (τ = 2.75 ms after brake onset), constant deceleration a = 2.951 m/s² slows the arm from 0.030 to 0.022 m/s, yet positive forward velocity continues compressing the spring to 67.6 N (vs. 72.0 N on the dashed red unbraked line).'],
 ['Compare the unbraked branch at the limit','At t = 8.33 ms, the dashed red unbraked trajectory hits the 100 N latch limit at full 0.030 m/s speed, whereas the green braked curve sits at 84.3 N (0.2108 mm compression) with arm speed reduced to 0.015 m/s.'],
 ['Stop at the modeled force limit','At t = 13.42 ms (10.17 ms after brake onset), arm speed reaches 0.000 m/s and spring compression peaks at 0.2500 mm right at the 100 N latch limit, consuming the entire 61 N braking budget with zero spare margin.']
 ].map(([label,description],i)=>({label,get description(){return root.dataset.lessonCommittedIndex===String(i)?description:'Next endpoint, not yet completed: '+description;}}));
 const force=t=>{const tau=Math.max(0,(t-D.onset_ms)/1000);return t<=D.onset_ms?D.k*D.v*t/1000:D.onset_force+D.k*(D.v*tau-.5*D.a*tau*tau);};
 function frame(t){const f=force(t),tau=Math.max(0,(t-D.onset_ms)/1000),v=Math.max(0,D.v-D.a*tau),nb=D.k*D.v*t/1000;
  actual.setAttribute('cx',X(t));actual.setAttribute('cy',Y(f));unbraked.style.display=nb<=145?'':'none';unbraked.setAttribute('cx',X(t));unbraked.setAttribute('cy',Y(nb));
  const points=Array.from({length:81},(_,i)=>{const z=t*i/80;return `${i?'L':'M'} ${X(z)} ${Y(force(z))}`;});trace.setAttribute('d',points.join(' '));
  const armX=2250+600*f/D.limit;arm.setAttribute('x',armX);const left=armX+180,right=3250,dx=(right-left)/10;spring.setAttribute('d',`M ${left} 1317 `+Array.from({length:9},(_,i)=>`L ${left+(i+1)*dx} ${1317+(i%2?35:-35)}`).join(' ')+` L ${right} 1317`);
  const mx=10+250*f/D.limit;miniArm.setAttribute('x',mx);const ml=mx+65,md=(460-ml)/10;miniSpring.setAttribute('d',`M ${ml} 44 `+Array.from({length:9},(_,i)=>`L ${ml+(i+1)*md} ${44+(i%2?12:-12)}`).join(' ')+` L 460 44`);physicalScene.setAttribute('aria-label',`Same modeled arm/plate: force ${f.toFixed(1)} N, compression ${(1000*f/D.k).toFixed(4)} mm, speed ${v.toFixed(3)} m/s; compression amplified`);
  root.dataset.compressionMm=String(1000*f/D.k);root.dataset.time=String(t);root.dataset.force=String(f);root.dataset.velocity=String(v);root.dataset.brakeOnset=String(t>=D.onset_ms);
  readout.textContent=`Since contact ${t.toFixed(2)} ms · braking-path force ${f.toFixed(1)} N · arm speed ${v.toFixed(3)} m/s · compression ${(1000*f/D.k).toFixed(4)} mm`;
  equation.textContent=t<D.onset_ms?`F = k v t; ${t<D.trip_ms?'tripwire not yet crossed':'tripwire crossed; response pending'}; physical deceleration has not begun.`:`F = 39 N + k(vτ − aτ²/2); τ = ${((t-D.onset_ms)).toFixed(2)} ms; a = ${D.a.toFixed(3)} m/s². Unbraked alternative ${nb.toFixed(1)} N${nb>145?' (above plot range)':''}.`;
 }
 function render(i,{stage,details}){root.dataset.lessonCommittedIndex=String(i);mount(stage,details);frame(times[i]);follow(X(times[i]),Y(force(times[i])));}
 function transition({from,to,onFrame}){if(to!==from+1)return;onFrame(p=>{frame(times[from]+(times[to]-times[from])*p);root.dataset.animationProgress=String(p);});}

 const player=window.MLSysOriginalPlayer.create({root,outer:context.outer,layout:'compact',title:"Why size the stop from force at brake onset?",intro:"Trace contact force from the 15 N tripwire across the 2 ms response window to 39 N at brake onset so the stop finishes at the 100 N latch limit.",scope:"Linear-spring and rigid-arm contact model (not a measured collision test): effective stiffness k = 400 N/mm (400,000 N/m), guarded approach speed v_0 = 0.030 m/s, pre-onset force rate k v_0 = 12 N/ms, contact tripwire F_trip = 15 N at 1.25 ms, response delay Δt_resp = 2.0 ms (detection + bus + torque rise), onset force F_onset = 39 N at 3.25 ms, required braking deceleration a = k v_0² / (2(F_lim - F_onset)) = 2.951 m/s² over 10.17 ms, and latch force limit F_lim = 100 N at 13.42 ms.",steps,render,transition});
 const destroy=player.destroy;player.destroy=()=>{disposed=true;cameraResize?.disconnect();cancelAnimationFrame(resizeFrame);destroy();};return player;
};})();

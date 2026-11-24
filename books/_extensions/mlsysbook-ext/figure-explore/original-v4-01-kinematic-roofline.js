/* Original-artwork Explore mechanism. Shared player owns the pausable clock. */
(()=>{'use strict';
window.MLSysFigureLessons=window.MLSysFigureLessons||{};
window.MLSysFigureLessons["original-v4-01-kinematic-roofline"]=async(root,context)=>{
 if(!context.verifiedArtwork||!window.MLSysOriginalPlayer)throw Error('Verified original artwork and shared player required');
 const svg=context.verifiedArtwork.cloneNode(true),NS='http://www.w3.org/2000/svg';
 const vb=svg.viewBox.baseVal,W=vb.width,H=vb.height;
 if(!W||!H)throw Error('Original artwork viewBox required');
 root.classList.add("original-v4-01-kinematic-roofline");
 svg.setAttribute('role','img');svg.setAttribute('aria-label',"Why does braking overtake delay travel above the 0.53 m/s crossover speed?");
 svg.querySelectorAll('text').forEach(n=>{const f=n.getAttribute('fill');if(f)n.style.setProperty('fill',f,'important');});
 const overlay=document.createElementNS(NS,'g');overlay.setAttribute('data-explore-overlay','');svg.append(overlay);
 const mk=(tag,attrs={},text)=>{const el=document.createElementNS(NS,tag);for(const[k,v]of Object.entries(attrs))el.setAttribute(k,String(v));if(text!=null)el.textContent=text;if(tag==='text'&&attrs.fill)el.style.setProperty('fill',String(attrs.fill),'important');return el;};
 const add=(tag,attrs={},text)=>{const el=mk(tag,attrs,text);overlay.append(el);return el;};
 add('path',{d:`M${(vb.x+W*.08).toFixed(1)},${(vb.y+H*.08).toFixed(1)}h${(W*.84).toFixed(1)}v${(H*.84).toFixed(1)}h-${(W*.84).toFixed(1)}z`,fill:'none',stroke:'none','data-mlsys-focus':''});
 const lerp=(a,b,t)=>a+(b-a)*t;
 const routePt=(pts,p)=>{if(pts.length===1)return pts[0];const lens=pts.slice(1).map((b,i)=>Math.hypot(b[0]-pts[i][0],b[1]-pts[i][1]));const tot=lens.reduce((s,x)=>s+x,0)||1;let rem=Math.max(0,Math.min(1,p))*tot,j=0;while(j<lens.length-1&&rem>lens[j]){rem-=lens[j];j++;}const q=lens[j]?rem/lens[j]:1;return[lerp(pts[j][0],pts[j+1][0],q),lerp(pts[j][1],pts[j+1][1],q)];};
 const canvas=document.createElement('div');canvas.className='mlsys-exp-canvas';canvas.tabIndex=0;canvas.setAttribute('aria-label','Original chapter diagram with synchronized state overlay.');canvas.append(svg);
 const readout=document.createElement('p');readout.className='mlsys-exp-readout';
 const invariant=document.createElement('p');invariant.className='mlsys-exp-invariant';
 const steps=[{"label": "Dominate with delay at 0.30 m/s", "description": "At low speed v = 0.30 m/s, linear travel during the 133.6 ms pre-brake delay is 40.1 mm while quadratic braking at 2.0 m/s² is only 22.5 mm (62.6 mm total stop)."}, {"label": "Cross equal terms at v* = 0.53 m/s", "description": "At the crossover speed v* = 2 a_brake τ_delay = 0.53 m/s (0.5344 m/s), delay travel (71.4 mm) and braking travel (71.4 mm) contribute equally for a 142.8 mm total stop."}, {"label": "Inspect half-speed scaling at 0.75 m/s", "description": "At half the drive limit (v = 0.75 m/s), delay travel reaches 100.2 mm while braking grows quadratically to 140.6 mm, yielding a 240.8 mm elementary stop."}, {"label": "Evaluate the 1.5 m/s drive limit", "description": "Doubling speed from 0.75 to the 1.5 m/s drive limit doubles delay travel to 200.4 mm (including 33.0 mm for the 22.0 ms backbone) but quadruples braking to 562.5 mm—totaling 762.9 mm with 337.1 mm unallocated spare below 1.10 m."}, {"label": "Reach the 1.10 m elementary ceiling", "description": "At v = 1.85 m/s (1.847 m/s), delay travel (246.8 mm) plus braking travel (853.2 mm) hits the 1,100.0 mm (1.10 m) clear distance—before localization, margin, and jerk allowances are even charged."}];

 // Verified PNG axes (3659 x 2460): x in [0, 2.0] -> [427, 3519], y in [0, 1600] -> [2133, 93]
 const X=v=>427+(3519-427)*(v/2.0),Y=mm=>2133-(2133-93)*(mm/1600.0);
 const tau=0.1336,a=2.0,speeds=[0.30,0.5344,0.75,1.50,1.8471];
 const vLine=add('line',{y1:93,y2:2133,stroke:'#007A87','stroke-width':10,'stroke-dasharray':'24 16','data-motion-object':'true'});
 const delayBar=add('line',{stroke:'#B87333','stroke-width':24,'stroke-linecap':'butt'});
 const brakeBar=add('line',{stroke:'#007A87','stroke-width':24,'stroke-linecap':'butt'});
 const dotDelay=add('circle',{r:26,fill:'#B87333',stroke:'#fff','stroke-width':7,'data-motion-object':'true'});
 const dotBrake=add('circle',{r:26,fill:'#007A87',stroke:'#fff','stroke-width':7,'data-motion-object':'true'});
 const dotTotal=add('circle',{r:32,fill:'#1A4D3E',stroke:'#fff','stroke-width':8,'data-motion-object':'true'});
 const callout=add('text',{y:220,'text-anchor':'middle','font-size':68,'font-weight':'700',fill:'#1A4D3E',stroke:'#fff','stroke-width':18,'paint-order':'stroke'},'');
 function paint(from,to,p){
  const v=lerp(speeds[from],speeds[to],p),idx=p<1?from:to;
  const dDelay=v*tau*1000,dBrake=(v*v)/(2*a)*1000,dTot=dDelay+dBrake,spare=1100-dTot;
  const x=X(v);
  vLine.setAttribute('x1',x);vLine.setAttribute('x2',x);
  delayBar.setAttribute('x1',x+26);delayBar.setAttribute('x2',x+26);delayBar.setAttribute('y1',Y(0));delayBar.setAttribute('y2',Y(dDelay));
  brakeBar.setAttribute('x1',x+26);brakeBar.setAttribute('x2',x+26);brakeBar.setAttribute('y1',Y(dDelay));brakeBar.setAttribute('y2',Y(dTot));
  dotDelay.setAttribute('cx',x);dotDelay.setAttribute('cy',Y(dDelay));
  dotBrake.setAttribute('cx',x);dotBrake.setAttribute('cy',Y(dBrake));
  dotTotal.setAttribute('cx',x);dotTotal.setAttribute('cy',Y(dTot));
  dotTotal.setAttribute('fill',spare<=2?'#A51C30':'#1A4D3E');
  callout.setAttribute('x',Math.min(2850,Math.max(1150,x)));
  callout.setAttribute('fill',spare<=2?'#A51C30':'#1A4D3E');
  callout.textContent=`v = ${v.toFixed(2)} m/s · stop ${dTot.toFixed(1)} mm (${spare>=0?spare.toFixed(1)+' mm spare':'at 1.10 m limit'})`;
  readout.textContent=`Speed v = ${v.toFixed(2)} m/s · delay travel v·τ = ${dDelay.toFixed(1)} mm + braking v²/(2a) = ${dBrake.toFixed(1)} mm = ${dTot.toFixed(1)} mm vs. 1,100.0 mm (1.10 m) clear distance`;
  invariant.textContent='Elementary roofline: d_stop(v) = v·(0.1336 s) + v²/(4.0 m/s²) · crossover v* = 0.53 m/s · 1 ms extra delay at 1.5 m/s costs 1.5 mm.';
  root.dataset.values=JSON.stringify({step:idx,speed_mps:+v.toFixed(4),delay_mm:+dDelay.toFixed(1),brake_mm:+dBrake.toFixed(1),total_mm:+dTot.toFixed(1)});
 }

 return window.MLSysOriginalPlayer.create({
  root,outer:context.outer,layout:'compact',
  title:"Why does braking overtake delay travel above the 0.53 m/s crossover speed?",
  intro:"Sweep speed from 0.30 m/s past the 0.53 m/s crossover and 1.5 m/s drive limit to see where total stop meets the 1.10 m clear distance.",
  scope:"mlsysim Scenarios.WarehouseAisleStopAtDriveLimit (new-explore-batch.calc.py): pre-brake delay \u03c4_delay = 133.6 ms, deceleration a_brake = 2.0 m/s\u00b2, clear distance D_clear = 1.10 m (1,100 mm), crossover v* = 0.53 m/s (0.5344 m/s), drive limit 1.5 m/s (762.9 mm), and elementary ceiling 1.85 m/s.",
  steps,
  render(i,{stage,details}){if(!canvas.isConnected)stage.append(canvas);if(details&&!readout.isConnected)details.append(readout,invariant);root.dataset.sceneStep=String(i);paint(i,i,1);},
  transition({from,to,onFrame}){if(to!==from+1)return;onFrame(p=>{root.dataset.animationProgress=String(p);paint(from,to,p);});}
 });
};})();

/* Original-artwork Explore mechanism. Shared player owns the pausable clock. */
(()=>{'use strict';
window.MLSysFigureLessons=window.MLSysFigureLessons||{};
window.MLSysFigureLessons["original-v4-18-indistinguishable-deadline"]=async(root,context)=>{
 if(!context.verifiedArtwork||!window.MLSysOriginalPlayer)throw Error('Verified original artwork and shared player required');
 const svg=context.verifiedArtwork.cloneNode(true),NS='http://www.w3.org/2000/svg';
 const vb=svg.viewBox.baseVal,W=vb.width,H=vb.height;
 if(!W||!H)throw Error('Original artwork viewBox required');
 root.classList.add("original-v4-18-indistinguishable-deadline");
 svg.setAttribute('role','img');svg.setAttribute('aria-label',"Why does late sensor separation fail to save a high-speed approach on hidden film?");
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
 const steps=[{"label": "Approach with identical observations at 1.30 m/s", "description": "In panels (a) and (b), both bases approach the 1.10 m harm boundary at v = 1.30 m/s while panel (c) shows identical teal sensor traces—no passive channel distinguishes dry concrete from clear hydraulic film yet."}, {"label": "Reach the action deadline unbeaten", "description": "At the vertical red action deadline (x = 435 in panel c), a state-specific brake would have to begin now, yet dry and film observation histories remain 100% identical."}, {"label": "Issue the same brake command", "description": "Crossing the action deadline commits the same brake command in both worlds (133.6 ms pre-brake delay plus 0.19 m localization, margin, and tracking overhead)."}, {"label": "Diverge too late: 0.969 m dry vs. 2.132 m oil", "description": "Wheel-slip traces in panel (c) finally separate inside the pink late-separation zone—after the deadline: dry floor (a = 2.0 m/s²) stops short at 0.969 m < 1.10 m, whereas hidden oil film (a = 0.49 m/s²) needs 2.132 m and overruns the harm boundary."}, {"label": "Enforce the 1.00 m/s inspected-floor restriction", "description": "Without a timely pre-deadline detector, the permission path must bound speed for the worst credible unseparated floor: cutting aisle speed 23.1% to 1.00 m/s on inspected floors (a = 1.18 m/s²) brings total stop down to 0.831 m < 1.10 m."}];

 const scanLine=add('line',{x1:140,x2:140,y1:296,y2:430,stroke:'#007A87','stroke-width':2.5,'stroke-dasharray':'5 3','data-motion-object':'true'});
 const dryCar=add('circle',{cx:95,cy:112,r:8,fill:'#1A4D3E',stroke:'#fff','stroke-width':2,'data-motion-object':'true'});
 const oilCar=add('circle',{cx:435,cy:112,r:8,fill:'#A51C30',stroke:'#fff','stroke-width':2,'data-motion-object':'true'});
 const badgeDry=add('text',{x:175,y:44,'text-anchor':'middle','font-size':10.5,'font-weight':'700',fill:'#1A4D3E',stroke:'#fff','stroke-width':3.2,'paint-order':'stroke'},'');
 const badgeOil=add('text',{x:515,y:44,'text-anchor':'middle','font-size':10.5,'font-weight':'700',fill:'#A51C30',stroke:'#fff','stroke-width':3.2,'paint-order':'stroke'},'');
 const scanXs=[220,435,485,575,435];
 const dryXs=[115,175,210,250,228];
 const oilXs=[455,515,565,638,568];
 const readouts=[
  'Pre-deadline approach (v = 1.30 m/s): y_dry(t) ≡ y_film(t) in panel (c) · state unseparated',
  'Action deadline t* reached: identical observation histories leave 0 bits to select a film-only brake early',
  'Common brake triggered (τ_delay = 133.6 ms, overhead = 0.19 m): physical friction now governs outcome',
  'Late separation (t > t*): dry stops at 0.969 m (< 1.10 m) · hidden oil film needs 2.132 m → OVERRUNS 1.10 m',
  'Qualified restriction: inspected floors + v = 1.00 m/s (−23.1%, a = 1.18 m/s²) → 0.831 m stop (< 1.10 m)'
 ];
 function paint(from,to,p){
  const idx=p<1?from:to;
  const sx=lerp(scanXs[from],scanXs[to],p),dx=lerp(dryXs[from],dryXs[to],p),ox=lerp(oilXs[from],oilXs[to],p);
  scanLine.setAttribute('x1',sx);scanLine.setAttribute('x2',sx);
  scanLine.setAttribute('stroke',idx===3?'#A51C30':idx===4?'#1A4D3E':'#007A87');
  dryCar.setAttribute('cx',dx);oilCar.setAttribute('cx',ox);
  oilCar.setAttribute('fill',idx===4?'#1A4D3E':'#A51C30');
  badgeDry.textContent=idx===4?'restricted 1.00 m/s: 0.831 m':idx>=3?'dry 1.30 m/s: 0.969 m (safe)':'v = 1.30 m/s';
  badgeOil.setAttribute('fill',idx===4?'#1A4D3E':'#A51C30');
  badgeOil.textContent=idx===4?'inspected floor: 0.831 m (safe)':idx>=3?'oil 1.30 m/s: 2.132 m (overrun!)':'v = 1.30 m/s (hidden film)';
  readout.textContent=readouts[idx];
  invariant.textContent='Observational indistinguishability: if y_safe(t) = y_hazard(t) through t*, permission must qualify one action for all unseparated states.';
  root.dataset.values=JSON.stringify({step:idx,d_dry_m:0.969,d_oil_m:2.132,d_restricted_m:0.831,d_clear_m:1.10});
 }

 return window.MLSysOriginalPlayer.create({
  root,outer:context.outer,layout:'compact',
  title:"Why does late sensor separation fail to save a high-speed approach on hidden film?",
  intro:"Advance both worlds past the action deadline at 1.30 m/s to see why dry stopping at 0.969 m and oil overrun at 2.132 m force a 1.00 m/s restriction.",
  scope:"mlsysim OilFilm & WarehouseRestrictedTransit scenarios (new-explore-batch.calc.py): D_clear = 1.10 m, \u03c4_delay = 133.6 ms, fixed overhead = 0.19 m; at v = 1.30 m/s dry concrete (a = 2.0 m/s\u00b2) requires 0.969 m while clear oil film (a = 0.49 m/s\u00b2) requires 2.132 m; inspected-floor restricted mode (v = 1.00 m/s, a = 1.18 m/s\u00b2) requires 0.831 m.",
  steps,
  render(i,{stage,details}){if(!canvas.isConnected)stage.append(canvas);if(details&&!readout.isConnected)details.append(readout,invariant);root.dataset.sceneStep=String(i);paint(i,i,1);},
  transition({from,to,onFrame}){if(to!==from+1)return;onFrame(p=>{root.dataset.animationProgress=String(p);paint(from,to,p);});}
 });
};})();

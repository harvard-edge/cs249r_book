/* Rollback bundles: toy request markers pinned at admission, drained before release. */
(()=>{'use strict';
const NAME='original-v1-rollback-compatible-bundles',NS='http://www.w3.org/2000/svg';
// Toy request ledger generated and checked by original-v1-rollback-compatible-bundles.calc.py.
const D={"requests":["R1","R2","R3","R4","R5","R6"],"states":[{"event":"start","target":2,"b1":0,"b2":4,"b2_state":"serving","b1_state":"standby","refused":null,"log":["R1","R2","R3","R4"],"log_flagged":true,"requests":{"R1":{"bundle":2,"status":"in-flight","admitted":"before"},"R2":{"bundle":2,"status":"in-flight","admitted":"before"},"R3":{"bundle":2,"status":"in-flight","admitted":"before"},"R4":{"bundle":2,"status":"in-flight","admitted":"before"},"R5":{"bundle":null,"status":"not-arrived","admitted":null},"R6":{"bundle":null,"status":"not-arrived","admitted":null}}},{"event":"cutover","target":1,"b1":0,"b2":4,"b2_state":"retained","b1_state":"standby","refused":null,"log":["R1","R2","R3","R4"],"log_flagged":true,"requests":{"R1":{"bundle":2,"status":"in-flight","admitted":"before"},"R2":{"bundle":2,"status":"in-flight","admitted":"before"},"R3":{"bundle":2,"status":"in-flight","admitted":"before"},"R4":{"bundle":2,"status":"in-flight","admitted":"before"},"R5":{"bundle":null,"status":"not-arrived","admitted":null},"R6":{"bundle":null,"status":"not-arrived","admitted":null}}},{"event":"admit","target":1,"b1":2,"b2":4,"b2_state":"retained","b1_state":"serving","refused":null,"log":["R1","R2","R3","R4"],"log_flagged":true,"requests":{"R1":{"bundle":2,"status":"in-flight","admitted":"before"},"R2":{"bundle":2,"status":"in-flight","admitted":"before"},"R3":{"bundle":2,"status":"in-flight","admitted":"before"},"R4":{"bundle":2,"status":"in-flight","admitted":"before"},"R5":{"bundle":1,"status":"in-flight","admitted":"after"},"R6":{"bundle":1,"status":"in-flight","admitted":"after"}}},{"event":"finish","target":1,"b1":2,"b2":2,"b2_state":"retained","b1_state":"serving","refused":null,"log":["R1","R2","R3","R4"],"log_flagged":true,"requests":{"R1":{"bundle":2,"status":"done","admitted":"before"},"R2":{"bundle":2,"status":"done","admitted":"before"},"R3":{"bundle":2,"status":"in-flight","admitted":"before"},"R4":{"bundle":2,"status":"in-flight","admitted":"before"},"R5":{"bundle":1,"status":"in-flight","admitted":"after"},"R6":{"bundle":1,"status":"in-flight","admitted":"after"}}},{"event":"refuse","target":1,"b1":2,"b2":2,"b2_state":"retained","b1_state":"serving","refused":"R3","log":["R1","R2","R3","R4"],"log_flagged":true,"requests":{"R1":{"bundle":2,"status":"done","admitted":"before"},"R2":{"bundle":2,"status":"done","admitted":"before"},"R3":{"bundle":2,"status":"in-flight","admitted":"before"},"R4":{"bundle":2,"status":"in-flight","admitted":"before"},"R5":{"bundle":1,"status":"in-flight","admitted":"after"},"R6":{"bundle":1,"status":"in-flight","admitted":"after"}}},{"event":"cancel_request","target":1,"b1":2,"b2":2,"b2_state":"retained","b1_state":"serving","refused":null,"log":["R1","R2","R3","R4"],"log_flagged":true,"requests":{"R1":{"bundle":2,"status":"done","admitted":"before"},"R2":{"bundle":2,"status":"done","admitted":"before"},"R3":{"bundle":2,"status":"in-flight","admitted":"before"},"R4":{"bundle":2,"status":"cancel-requested","admitted":"before"},"R5":{"bundle":1,"status":"in-flight","admitted":"after"},"R6":{"bundle":1,"status":"in-flight","admitted":"after"}}},{"event":"cancel_confirm","target":1,"b1":2,"b2":1,"b2_state":"retained","b1_state":"serving","refused":null,"log":["R1","R2","R3","R4"],"log_flagged":true,"requests":{"R1":{"bundle":2,"status":"done","admitted":"before"},"R2":{"bundle":2,"status":"done","admitted":"before"},"R3":{"bundle":2,"status":"in-flight","admitted":"before"},"R4":{"bundle":2,"status":"cancel-confirmed","admitted":"before"},"R5":{"bundle":1,"status":"in-flight","admitted":"after"},"R6":{"bundle":1,"status":"in-flight","admitted":"after"}}},{"event":"finish","target":1,"b1":2,"b2":0,"b2_state":"releasable","b1_state":"serving","refused":null,"log":["R1","R2","R3","R4"],"log_flagged":true,"requests":{"R1":{"bundle":2,"status":"done","admitted":"before"},"R2":{"bundle":2,"status":"done","admitted":"before"},"R3":{"bundle":2,"status":"done","admitted":"before"},"R4":{"bundle":2,"status":"cancel-confirmed","admitted":"before"},"R5":{"bundle":1,"status":"in-flight","admitted":"after"},"R6":{"bundle":1,"status":"in-flight","admitted":"after"}}},{"event":"release","target":1,"b1":2,"b2":0,"b2_state":"released","b1_state":"serving","refused":null,"log":["R1","R2","R3","R4"],"log_flagged":true,"requests":{"R1":{"bundle":2,"status":"done","admitted":"before"},"R2":{"bundle":2,"status":"done","admitted":"before"},"R3":{"bundle":2,"status":"done","admitted":"before"},"R4":{"bundle":2,"status":"cancel-confirmed","admitted":"before"},"R5":{"bundle":1,"status":"in-flight","admitted":"after"},"R6":{"bundle":1,"status":"in-flight","admitted":"after"}}}]};
// Drawing coordinates in the verified SVG; calc.py checks source rectangles and token clearances.
const G={"source":{"router":[0.6,0.2,356.5,22.8],"bundle1":[11.9,46.1,130.5,84.2],"bundle2":[226.6,46.1,345.2,84.2],"green_x":71.2,"dashed_x":285.9,"invalid":[138.8,46.1,218.3,84.2],"log":[2.1,124.0,355.0,144.8]},"slots":{"R1":[248,28],"R2":[267,28],"R3":[248,37],"R4":[267,37],"R5":[90,28],"R6":[109,28]},"token":[17,8.5]};
const DESK={tok:[...G.token,5.6],chip:{y:11.5,w:45,h:11,f:4.8,x:[null,26.5,331]},badge:{1:[103,46.5],2:[317,46.5],w:52,h:9,f:5.3},ghost:[178.5,33],refused:[178.5,94,5.6],
 arrowTop:{1:[G.source.green_x,23],2:[G.source.dashed_x,23]},slots:G.slots,dim:[[27,23.3,47,21]],invalid:G.source.invalid,b2:G.source.bundle2,log:G.source.log,logText:[40,136.4,5.6],flag:[320,134.4,56,12,5.4]};
const MOB={tok:[28,16,10.5],chip:{y:47,w:104,h:15,f:10,x:[null,72,218]},badge:{1:[100,130],2:[190,130],w:80,h:16,f:10},ghost:[34,305],refused:[145,327,11],
 arrowTop:{1:[52,60],2:[238,60]},slots:{R1:[170,82],R2:[204,82],R3:[170,104],R4:[204,104],R5:[80,84],R6:[114,84]},dim:[[2,60,56,64]],invalid:[2,282,286,52],b2:[148,130,140,94],log:[2,342,286,60],logText:[50,395,10],flag:[236,391,92,15,10]};
const el=(tag,a={},text)=>{const n=document.createElementNS(NS,tag);for(const[k,v]of Object.entries(a))n.setAttribute(k,v);if(text!=null)n.textContent=text;return n;};
const FONT='Helvetica, Arial, sans-serif';
// Responsive redraw of the same source objects, labels, and arrow semantics for phones.
function mobileArt(){
 const s=el('svg',{viewBox:'0 0 290 406',role:'img','aria-label':'Responsive version of the same router, two bundles, invalid pairing, and degraded-window log box'});s.classList.add('rb-mobile');
 const t=(x,y,v,a={})=>s.append(el('text',{x,y,'font-family':FONT,'font-size':11,fill:'#000','text-anchor':'middle',...a},v));
 s.append(el('rect',{x:1,y:1,width:288,height:58,rx:3,fill:'#f6f6f6',stroke:'#666','stroke-width':1}));
 t(145,18,'Router pins model, feature schema, and',{'font-size':11.5});t(145,33,'embedding namespace at admission',{'font-size':11.5});
 s.append(el('path',{d:'M52,59V122',stroke:'#008f45','stroke-width':1.6,fill:'none'}),el('path',{d:'M48,121L52,129L56,121Z',fill:'#008f45'}));
 s.append(el('path',{d:'M238,59V122',stroke:'#666','stroke-width':1.6,'stroke-dasharray':'5 4',fill:'none'}),el('path',{d:'M234,121L238,129L242,121Z',fill:'#666'}));
 ['new','requests','after','rollback'].forEach((v,i)=>t(47,78+i*12,v,{'text-anchor':'end','font-size':10,fill:'#008f45'}));
 ['already','admitted','requests','stay','pinned'].forEach((v,i)=>t(243,74+i*12,v,{'text-anchor':'start','font-size':10}));
 [[2,1,'previous release'],[148,2,'candidate release']].forEach(([x,n,role])=>{s.append(el('rect',{x,y:130,width:140,height:94,rx:3,fill:'#eaf9ff',stroke:'#00638f','stroke-width':1.4}));const c=x+70;
  t(c,154,'Bundle '+n+':',{'font-size':12,'font-weight':700});t(c,169,role,{'font-size':12,'font-weight':700});t(c,186,`weights ${n} · schema ${n}`);t(c,201,'embedding/cache');t(c,215,'namespace '+n);});
 ['Switch the routing target','to the complete','compatible bundle.'].forEach((v,i)=>t(72,242+i*13,v,{'font-size':10}));
 ['Retain until requests','finish or their','cancellation is confirmed.'].forEach((v,i)=>t(218,242+i*13,v,{'font-size':10}));
 s.append(el('rect',{x:2,y:282,width:286,height:52,fill:'#fef5f7',stroke:'#d9534f','stroke-width':1,'stroke-dasharray':'5 4'}));
 t(145,299,'Invalid pairing:',{fill:'#c9302c'});t(145,313,'weights 1 with namespace 2',{fill:'#c9302c'});
 s.append(el('rect',{x:2,y:342,width:286,height:60,rx:3,fill:'#fef5f7',stroke:'#d9534f','stroke-width':1}));
 t(145,360,'Degraded-window logs: quarantine or repair',{'font-size':11});t(145,375,'before retraining',{'font-size':11});
 return s;
}
const STATUS={'in-flight':'in flight','done':'finished','cancel-requested':'cancel requested, still running','cancel-confirmed':'cancellation confirmed','not-arrived':'not yet arrived'};
window.MLSysFigureLessons=window.MLSysFigureLessons||{};
window.MLSysFigureLessons[NAME]=async(root,{outer,verifiedArtwork})=>{
 if(!verifiedArtwork||!window.MLSysOriginalPlayer)throw Error('Verified rollback artwork and shared player required');
 if(verifiedArtwork.getAttribute('viewBox')!=='0 0 357.564 145.496')throw Error('Rollback artwork geometry changed; review the mappings');
 root.classList.add(NAME);
 const art=verifiedArtwork.cloneNode(true);art.removeAttribute('width');art.removeAttribute('height');art.classList.add('rb-original');
 art.setAttribute('role','img');art.setAttribute('aria-label','Original rollback figure with toy request markers, router target, in-flight counts, and release state');
 const deskLayer=el('g',{'data-rb-overlay':'desktop'});art.append(deskLayer);
 const mob=mobileArt(),mobLayer=el('g',{'data-rb-overlay':'mobile'});mob.append(mobLayer);
 const source=document.createElement('div');source.className='rb-source';source.append(art,mob);
 const summary=document.createElement('p');summary.className='rb-summary';
 const table=document.createElement('table');table.className='rb-ledger';
 table.innerHTML='<thead><tr><th scope="col">Request</th><th scope="col">Admitted</th><th scope="col">Pinned to</th><th scope="col">Status</th></tr></thead><tbody></tbody>';
 const body=table.tBodies[0],toy=document.createElement('p');toy.className='rb-toy';toy.textContent='Toy request markers R1–R6 (illustrative: the chapter gives no request counts, rates, or durations).';table.setAttribute('aria-label','Toy request ledger');
 // Each label/description is one causal change; counters in D derive from the per-request ledger.
 const text=[
  ['Bundle 2 is the routing target','Before rollback in this 6-request trace (R1–R6), the router admits R1–R4 to candidate Bundle 2 (weights 2, schema 2, namespace 2; 4 in flight), while Bundle 1 stays loaded on standby (0 in flight). Records from R1–R4 fall in the degraded window.'],
  ['Cutover selects Bundle 1','Rollback changes one routing decision covering weights 1, schema 1, and namespace 1 together. No request moves: R1–R4 stay pinned to Bundle 2, so its in-flight count is still 4.'],
  ['New requests pin to Bundle 1','R5 and R6 arrive after rollback and take the solid green path. Both bundles now serve at once: new requests on Bundle 1, earlier requests on Bundle 2.'],
  ['Pinned requests finish on Bundle 2','R1 and R2 complete in the vector space they started in. Bundle 2’s in-flight count drops from 4 to 2; it must stay loaded for R3 and R4.'],
  ['Mixed pairing is refused','Moving R3 to weights 1 while it keeps namespace 2 would score its queries against incompatible vectors. The pairing is invalid, so the move is refused and R3 stays on Bundle 2.'],
  ['Cancellation of R4 is requested','Asking to cancel R4 does not end it. R4 still counts against Bundle 2 until its cancellation is confirmed, so the count stays at 2.'],
  ['R4 cancellation is confirmed','Confirmation ends R4’s use of Bundle 2. The in-flight count drops to 1: R3 is now the last consumer of Bundle 2.'],
  ['The last pinned request finishes','R3 completes and Bundle 2’s in-flight count reaches 0. Only now may Bundle 2 be released. Bundle 1 serves all traffic.'],
  ['Release Bundle 2; logs stay flagged','Bundle 2 is released, but the R1–R4 records from the degraded window remain in the logs and must be quarantined or repaired before retraining. Cost: both bundles stayed loaded through the drain, and reverting routing repaired no data.']
 ];
 if(text.length!==D.states.length)throw Error('Rollback narration and ledger disagree');
 let committed=0,mounted=false;
 const steps=text.map(([label,description],i)=>({label,get description(){return committed===i?description:'On completion: '+description;}}));
 const live=s=>s==='in-flight'||s==='cancel-requested';
 // Token set at a committed state: live requests, plus the refused R3 ghost.
 function tokens(i,L){const st=D.states[i],out={};for(const r of D.requests){const q=st.requests[r];if(q.bundle&&live(q.status))out[r]={xy:L.slots[r],bundle:q.bundle,cancel:q.status==='cancel-requested'};}
  if(st.refused)out.ghost={xy:L.ghost,label:st.refused,ghost:true};return out;}
 function token(g,L,id,t,xy,opacity){const[w,h,f]=L.tok,b=t.bundle===1,c=t.ghost?['#fdebec','#c9302c','#8f1d1a']:b?['#dff3e6','#008f45','#0b4d2a']:['#eef0f2','#4d5660','#17212b'];
  const n=el('g',{transform:`translate(${xy[0]} ${xy[1]})`,opacity,'data-token':id});
  n.append(el('rect',{x:-w/2,y:-h/2,width:w,height:h,rx:h/2.6,fill:c[0],stroke:c[1],'stroke-width':h/9,'stroke-dasharray':t.cancel||t.ghost?`${h/4} ${h/6}`:'none'}),
   el('text',{x:0,y:f*.36,'text-anchor':'middle','font-family':FONT,'font-size':f,'font-weight':700,fill:c[2]},(t.label||id)));g.append(n);}
 function badge(g,L,x,y,label,kind){const{w,h,f}=L.badge,c=kind==='released'?['#e6f4ea','#008f45','#0b4d2a']:kind==='zero'?['#fff7e0','#a36b00','#5c3d00']:['#ffffff','#00638f','#0b3a52'];
  g.append(el('rect',{x:x-w/2,y:y-h/2,width:w,height:h,rx:h/3,fill:c[0],stroke:c[1],'stroke-width':h/12}),el('text',{x,y:y+f*.36,'text-anchor':'middle','font-family':FONT,'font-size':f,'font-weight':700,fill:c[2]},label));}
 // Paint one layout. p<1 interpolates movement from state `from`; committed counters always use `to`'s predecessor until p reaches 1.
 function paint(g,L,from,to,p){g.replaceChildren();const st=D.states[p>=1?to:from],fin=p>=1;
  if(D.states[fin?to:from].target===2)L.dim.forEach(([x,y,w,h])=>g.append(el('rect',{x,y,width:w,height:h,fill:'#fff',opacity:.72})));
  // Refused pairing lights the source invalid box while the ghost approaches it.
  const refusing=D.states[to].refused&&(fin||p>0);if(refusing){const[x,y,a,b]=L===DESK?[L.invalid[0],L.invalid[1],L.invalid[2]-L.invalid[0],L.invalid[3]-L.invalid[1]]:L.invalid;g.append(el('rect',{x,y,width:a,height:b,fill:'#d9534f','fill-opacity':.14*(fin?1:p),stroke:'#c9302c','stroke-width':L===DESK?1.2:2.4}));}
  if(fin&&st.refused){const[x,y,f]=L.refused;g.append(el('text',{x,y,'text-anchor':'middle','font-family':FONT,'font-size':f,'font-weight':700,fill:'#a12622'},st.refused+' move refused; stays on Bundle 2'));}
  if(st.b2_state==='released'){const[x,y,w,h]=L===DESK?[L.b2[0],L.b2[1],L.b2[2]-L.b2[0],L.b2[3]-L.b2[1]]:L.b2;g.append(el('rect',{x,y,width:w,height:h,fill:'#fff',opacity:.55}));}
  // Router target chip: one decision for the whole bundle.
  const cx=L.chip.x[D.states[from].target]+(L.chip.x[D.states[to].target]-L.chip.x[D.states[from].target])*p,tgt=fin?D.states[to].target:D.states[from].target;
  g.append(el('rect',{x:cx-L.chip.w/2,y:L.chip.y-L.chip.h/2,width:L.chip.w,height:L.chip.h,rx:L.chip.h/2,fill:'#1f407a'}),el('text',{x:cx,y:L.chip.y+L.chip.f*.36,'text-anchor':'middle','font-family':FONT,'font-size':L.chip.f,'font-weight':700,fill:'#fff'},'target: Bundle '+tgt));
  badge(g,L,...L.badge[1],st.b1?`in flight ${st.b1}`:'standby · 0');
  badge(g,L,...L.badge[2],st.b2_state==='released'?'released':st.b2===0?'0 · releasable':`in flight ${st.b2}`,st.b2_state==='released'?'released':st.b2===0?'zero':'');
  const[lx,ly,lf]=L.logText;g.append(el('text',{x:lx,y:ly,'text-anchor':'middle','font-family':FONT,'font-size':lf,'font-weight':700,fill:'#a12622'},'logged: '+st.log[0]+'–'+st.log[st.log.length-1]));
  if(st.b2_state==='released'){const[x,y,w,h,f]=L.flag,[bx,by,bw,bh]=L===DESK?[L.log[0],L.log[1],L.log[2]-L.log[0],L.log[3]-L.log[1]]:L.log;g.append(el('rect',{x:bx,y:by,width:bw,height:bh,rx:3,fill:'none',stroke:'#c9302c','stroke-width':L===DESK?1.3:2.6}),el('rect',{x:x-w/2,y:y-h/2,width:w,height:h,rx:h/3,fill:'#a12622'}),el('text',{x,y:y+f*.36,'text-anchor':'middle','font-family':FONT,'font-size':f,'font-weight':700,fill:'#fff'},'still flagged'));}
  // Tokens: same identity across states; admissions travel down their source arrow.
  const A=tokens(from,L),B=tokens(to,L),arrivals=Object.keys(B).filter(id=>!A[id]&&!B[id].ghost);// Admissions leave the router one after another, never stacked.
  
  for(const id of new Set([...Object.keys(A),...Object.keys(B)])){const a=A[id],b=B[id];
   if(fin){if(b)token(g,L,id,b,b.xy,1);continue;}
   if(a&&b)token(g,L,id,a,[a.xy[0]+(b.xy[0]-a.xy[0])*p,a.xy[1]+(b.xy[1]-a.xy[1])*p],1);
   else if(a)token(g,L,id,a,a.xy,1-p);
   else{const k=b.ghost?0:arrivals.indexOf(id),o=b.ghost?L.slots[b.label]:L.arrowTop[b.bundle],q=b.ghost?Math.min(1,p*1.25):Math.max(0,Math.min(1,(p-.35*k)/.65));if(!b.ghost&&q<=0)continue;token(g,L,id,b,[o[0]+(b.xy[0]-o[0])*q,o[1]+(b.xy[1]-o[1])*q],b.ghost?.35+.65*p:Math.min(1,q*3));}}
 }
 function readout(i){const st=D.states[i];
  summary.textContent=`Router target: Bundle ${st.target} · Bundle 1 in flight: ${st.b1}${st.b1?'':' (standby)'} · Bundle 2 in flight: ${st.b2} (${st.b2_state}) · Degraded-window records: ${st.log.length} (${st.log.join(', ')}), ${st.b2_state==='released'?'still flagged after rollback':'flagged'}`+(st.refused?` · ${st.refused} → weights 1 + namespace 2: refused`:'');
  body.replaceChildren(...D.requests.map(r=>{const q=st.requests[r],tr=document.createElement('tr');tr.dataset.status=q.status;
   [r,q.admitted?q.admitted+' cutover':'—',q.bundle?'Bundle '+q.bundle:'—',STATUS[q.status]].forEach((v,k)=>{const c=document.createElement(k?'td':'th');if(!k)c.scope='row';c.textContent=v;tr.append(c);});return tr;}));
  root.dataset.ledger=JSON.stringify({target:st.target,b1_in_flight:st.b1,b2_in_flight:st.b2,b2_state:st.b2_state,refused:st.refused,log_records:st.log.length,log_flagged:st.log_flagged,requests:Object.fromEntries(D.requests.map(r=>[r,st.requests[r].status]))});
 }
 const player=window.MLSysOriginalPlayer.create({root,outer,layout:'compact',intro:'Trace 6 requests across a router rollback from Bundle 2 to Bundle 1 to see why Bundle 2 cannot be released until all 4 pinned requests drain.',title:'When can the candidate bundle be released after rollback?',
  scope:'The original rollback figure with six toy request markers, R1–R6. Their number, order, and completion pattern are illustrative: the chapter gives no request counts, rates, or durations, and movement speed encodes none. No drain timeout is modeled. Degraded-window records are counted for requests admitted to Bundle 2.',steps,
  render(i,{stage,details}){committed=i;if(!mounted){stage.append(source);(details||stage).append(summary,toy,table);mounted=true;}paint(deskLayer,DESK,i,i,1);paint(mobLayer,MOB,i,i,1);readout(i);delete root.dataset.pending;},
  transition({from,to,onFrame}){root.dataset.pending=String(to);onFrame(p=>{paint(deskLayer,DESK,from,to,p);paint(mobLayer,MOB,from,to,p);root.dataset.animationProgress=String(p);});}
 });
 return player;
};})();

/* Fairness threshold: a shared boundary moves across the original two-subgroup
 * drawing from the 75% line to the 81.25% line, then Subgroup B gets its own
 * cutoff. The boundary segments and approval shading follow the interpolated
 * positions; error rings, counts, and the ledger describe only the committed step. */
(()=>{'use strict';
const NAME='original-v1-fairness-threshold',NS='http://www.w3.org/2000/svg';
// Generated and checked by original-v1-fairness-threshold.calc.py from the figure's TikZ.
// pos is the fractional TikZ position along a subgroup row; x/cy are verified-SVG coordinates.
// stops[i] is [A boundary, B boundary]; they are equal while the boundary is shared.
const D={"viewBox":"0 0 292.214 87.545","stops":[null,[0.63,0.63],[0.63,0.63],[0.54,0.54],[0.5,0.5],[0.465,0.465],[0.415,0.415],[0.3,0.3],[0.3,0.3],[0.3,0.625],[0.3,0.725],[0.3,0.795]],"phases":["none","decide","score","score","score","score","score","score","compare","split","split","split"],"mids":[0.54,0.5,0.465,0.415],"bStops":[0.625,0.725,0.795],"bounds":[0.3,0.63],"labels":["81.25","75"],"tx":42.193274,"k":0.696381,"u0":-0.00268154,"u1":354.334986,"rowY":[31.7695,56.4453],"lineY":[12.0273,63.7461],"splitY":37.1582,"markers":[{"g":0,"y":1,"pos":0.05,"x":54.5313,"cy":23.873},{"g":0,"y":1,"pos":0.12,"x":71.8027,"cy":23.873},{"g":0,"y":1,"pos":0.19,"x":89.0762,"cy":23.873},{"g":0,"y":1,"pos":0.26,"x":106.3477,"cy":23.873},{"g":0,"y":0,"pos":0.38,"x":135.959,"cy":23.873},{"g":0,"y":0,"pos":0.45,"x":153.2285,"cy":23.873},{"g":0,"y":0,"pos":0.52,"x":170.5039,"cy":23.873},{"g":0,"y":0,"pos":0.8,"x":239.5938,"cy":23.873},{"g":1,"y":1,"pos":0.08,"x":61.9336,"cy":48.5488},{"g":1,"y":1,"pos":0.48,"x":160.6329,"cy":48.5488},{"g":1,"y":1,"pos":0.56,"x":180.3731,"cy":48.5488},{"g":1,"y":1,"pos":0.76,"x":229.7247,"cy":48.5488},{"g":1,"y":0,"pos":0.69,"x":212.4512,"cy":48.5469},{"g":1,"y":0,"pos":0.83,"x":246.9941,"cy":48.5469},{"g":1,"y":0,"pos":0.9,"x":264.2656,"cy":48.5469},{"g":1,"y":0,"pos":0.97,"x":281.5391,"cy":48.5469}]};
const INK='#17212b',APPROVE='#008f45',GROUP=['Subgroup A','Subgroup B'];
const sx=pos=>D.tx+D.k*(D.u0+pos*(D.u1-D.u0)),X0=sx(0);
const svg=(tag,a={},text)=>{const n=document.createElementNS(NS,tag);for(const[k,v]of Object.entries(a))n.setAttribute(k,v);if(text!=null)n.textContent=text;return n;};
// Same equations as the calc script: counts of drawn markers left of (approved by) their row's boundary.
const pct=(n,d)=>(100*n/d).toFixed(2).replace(/(\.\d*?)0+$/,'$1').replace(/\.$/,'');
const rate=(n,d)=>(n/d).toFixed(2);
const LABEL={[D.bounds[0]]:D.labels[0],[D.bounds[1]]:D.labels[1]};
const nameOf=b=>b==null?null:b[0]===b[1]?(b[0] in LABEL?LABEL[b[0]]+'% line':D.mids.includes(b[0])?'Move '+(D.mids.indexOf(b[0])+1):null):'B cutoff '+(D.bStops.indexOf(b[1])+1);
function ledger(i,phase=D.phases[i],b=D.stops[i]){
 const L={step:i,phase,boundary:b,name:nameOf(b)};
 if(b==null)return L;
 const ok=m=>m.pos<b[m.g],count=f=>[0,1].map(g=>D.markers.filter(m=>m.g===g&&f(m)).length);
 L.approved=count(ok);if(phase==='decide')return L;
 L.repayersApproved=count(m=>m.y===1&&ok(m));L.defaultersApproved=count(m=>m.y===0&&ok(m));
 L.deniedRepayers=L.repayersApproved.map(v=>4-v);L.errors=[0,1].map(g=>L.deniedRepayers[g]+L.defaultersApproved[g]);
 L.wrong=D.markers.map((m,j)=>ok(m)!==(m.y===1)?j:-1).filter(j=>j>=0);
 L.correct=16-L.errors[0]-L.errors[1];L.accuracyPct=pct(L.correct,16);
 L.tpr=L.repayersApproved.map(v=>rate(v,4));L.fpr=L.defaultersApproved.map(v=>rate(v,4));
 return L;
}
const ledgers=D.stops.map((_,i)=>ledger(i)),FIRST=ledgers[2],LINE=ledgers[7],CMP=ledgers[8],LAST=ledgers[ledgers.length-1];
const pick=(L,keys)=>Object.fromEntries(keys.map(k=>[k,L[k]])),KEYS=['name','errors','correct','accuracyPct','deniedRepayers','defaultersApproved','tpr','fpr'];
CMP.before=pick(FIRST,KEYS);LAST.shared=pick(LINE,KEYS);
// Every distinct shared cutoff (same sweep as the calc): how many reach the best accuracy, and the next best.
{const P=[...new Set(D.markers.map(m=>m.pos))].sort((a,b)=>a-b),cuts=[P[0]-0.01,...P.slice(1).map((v,j)=>(P[j]+v)/2),P[P.length-1]+0.01];
 const C=cuts.map(c=>ledger(-1,'score',[c,c]).correct),MAX=Math.max(...C),NEXT=Math.max(...C.filter(v=>v<MAX));
 CMP.sweep={cutoffs:cuts.length,bestCorrect:MAX,bestPct:pct(MAX,16),bestCount:C.filter(v=>v===MAX).length,nextPct:pct(NEXT,16),correctByPosition:C};
}
// 1-based sweep position of each candidate line: shared position j sits right of j-1 drawn markers.
const LINE_POS=D.bounds.map(x=>D.markers.filter(m=>m.pos<x).length+1);
const n=(v,word)=>v+' '+word+(v===1?'':'s');
const kinds=(dr,fa)=>[dr?(dr===1?'a denied repayer':dr+' denied repayers'):'',fa?(fa===1?'an approved defaulter':fa+' approved defaulters'):''].filter(Boolean).join(' and ')||'no errors';
// Narration derives its numbers from the committed ledgers; none are typed here.
const prev=L=>ledgers[L.step-1];
const states=[
 ['Sixteen toy applicants, two candidate boundaries',()=>'Each row is one subgroup of eight drawn applicants: circles repaid, plus markers defaulted. Horizontal position is the model\'s score order: markers further left have higher scores, so a boundary approves everything to its left, and moving a boundary right is more lenient. The two dashed lines are candidate shared boundaries, each labeled with the overall accuracy it achieves, not a score cutoff. No boundary is applied yet.'],
 ['Apply the 75% boundary to both rows',L=>`One shared boundary approves every applicant to its left, in both rows. It approves ${L.approved[0]} of 8 in Subgroup A and ${L.approved[1]} of 8 in Subgroup B. Decisions are made; they have not yet been checked against outcomes.`],
 ['Check the decisions against outcomes',L=>`A denied repayer or an approved defaulter is a misclassification (dashed ring with ✗). Subgroup A has ${kinds(L.deniedRepayers[0],L.defaultersApproved[0])}; Subgroup B has ${kinds(L.deniedRepayers[1],L.defaultersApproved[1])}. Overall ${L.correct} of 16 are correct: ${L.accuracyPct}%, the dashed line's label.`],
 ['Move 1: the boundary passes a Subgroup B repayer',L=>`Moving left makes the shared boundary stricter. It now denies a Subgroup B applicant who repaid, so B rises to ${n(L.errors[1],'error')} while A stays at ${L.errors[0]}. Accuracy falls to ${L.accuracyPct}%.`],
 ['Move 2: the boundary passes a Subgroup A defaulter',L=>`Denying a Subgroup A defaulter is a correct decision, so A falls to ${n(L.errors[0],'error')}; B stays at ${L.errors[1]}. Accuracy returns to ${L.accuracyPct}%, the same as at the ${FIRST.name}, with a different error split.`],
 ['Move 3: the boundary passes another Subgroup B repayer',L=>`Another Subgroup B repayer is denied: B rises to ${n(L.errors[1],'error')}, A stays at ${L.errors[0]}. Accuracy falls to ${L.accuracyPct}% again. Every B marker the stricter boundary has crossed so far was a repayer.`],
 ['Move 4: the boundary passes another Subgroup A defaulter',L=>`A falls to ${n(L.errors[0],'error')} and B stays at ${L.errors[1]}. Accuracy is back at ${L.accuracyPct}%: ${L.correct} of 16, as at the ${FIRST.name}, but now most errors are in Subgroup B.`],
 ['Reach the 81.25% line',L=>`The boundary passes Subgroup A's last approved defaulter. A is perfectly separated (${n(L.errors[0],'error')}); B keeps ${L.errors[1]}, all denied repayers. Overall ${L.correct} of 16 are correct: ${L.accuracyPct}%, the dashed line's label.`],
 ['Compare the candidates and check every shared position',L=>{const b=L.before,s=L.sweep;
  return `Moving from the ${b.name} to the ${L.name} lifts accuracy from ${b.accuracyPct}% to ${L.accuracyPct}%. Every gain falls in Subgroup A (${b.errors[0]} to ${n(L.errors[0],'error')}); Subgroup B goes from ${n(b.errors[1],'misclassification')} to ${L.errors[1]}, all denied repayers. Checking all ${s.cutoffs} shared positions (listed one by one below the ledger), ${s.bestPct}% is reached ${s.bestCount===1?'only here':'at '+s.bestCount+' positions'}; every other shared position scores at most ${s.nextPct}%. No shared boundary does better, yet at the best one Subgroup B approves ${L.repayersApproved[1]} of its 4 repayers (TPR ${L.tpr[1]}) against Subgroup A's ${L.repayersApproved[0]} of 4 (TPR ${L.tpr[0]}).`;}],
 ['Give Subgroup B its own cutoff',L=>{const P=prev(L);return `Subgroup A stays on the ${D.labels[0]}% line. Subgroup B's cutoff moves right, a more lenient threshold for B alone, and approves the two repayers B lost in Moves 1 and 3. B falls from ${P.errors[1]} to ${n(L.errors[1],'error')} (${kinds(L.deniedRepayers[1],L.defaultersApproved[1])}); B's TPR rises from ${P.tpr[1]} to ${L.tpr[1]} and its FPR stays at ${L.fpr[1]}. Overall accuracy is already ${L.accuracyPct}% (${L.correct} of 16), above the best shared boundary's ${CMP.sweep.bestPct}%, with no false positive. TPR is not yet equal: B's ${L.tpr[1]} against A's ${L.tpr[0]}. This is no longer one shared boundary.`;}],
 ['B\'s cutoff approves a defaulter',L=>`Moving B's cutoff further right approves a Subgroup B applicant who defaulted. B's FPR rises from ${prev(L).fpr[1]} to ${L.fpr[1]} while its TPR stays at ${L.tpr[1]}; B now has ${n(L.errors[1],'error')} (${kinds(L.deniedRepayers[1],L.defaultersApproved[1])}) and overall accuracy falls to ${L.accuracyPct}%. B's last denied repayer sits right of this defaulter, so B's cutoff cannot equalize TPR without approving it.`],
 ['B\'s cutoff equalizes TPR, at a cost',L=>{const S=L.shared,C1=ledgers[CMP.step+1];
  return `B's cutoff now approves its last repayer. B's TPR is ${L.tpr[1]}, equal to A's ${L.tpr[0]}, but B's FPR is ${L.fpr[1]} against A's ${L.fpr[0]}: the one remaining B error is ${kinds(L.deniedRepayers[1],L.defaultersApproved[1])}, where the shared ${S.name} left ${kinds(S.deniedRepayers[1],S.defaultersApproved[1])}. Overall accuracy is ${L.accuracyPct}% (${L.correct} of 16), ${C1.accuracyPct===L.accuracyPct?'the same as':'against '+C1.accuracyPct+'%'} at ${C1.name}, where B's FPR was ${C1.fpr[1]} but its TPR only ${C1.tpr[1]}. The false positive is the price of equal TPR, not of a separate cutoff, and accuracy alone cannot choose between the two. This is the threshold adjustment the text describes, with its trade-off. Group-specific cutoffs are prohibited in US employment selection and restricted in credit; whether this trade is acceptable is a policy decision, not a metric.`;}]
];
if(states.length!==ledgers.length)throw Error('Narration and ledgers disagree on the number of steps');

window.MLSysFigureLessons=window.MLSysFigureLessons||{};
window.MLSysFigureLessons[NAME]=async(root,{outer,verifiedArtwork})=>{
 if(!verifiedArtwork||!window.MLSysOriginalPlayer)throw Error('Verified fairness-threshold diagram and shared player required');
 if(verifiedArtwork.getAttribute('viewBox')!==D.viewBox)throw Error('Fairness-threshold artwork geometry changed; review the mappings');
 root.classList.add(NAME);
 const art=verifiedArtwork.cloneNode(true);art.removeAttribute('width');art.removeAttribute('height');
 art.setAttribute('data-mlsys-keep','');art.setAttribute('role','img');art.setAttribute('aria-label','Original two-subgroup score diagram. A shared decision boundary moves from the 75 percent line to the 81.25 percent line, then Subgroup B receives its own cutoff; misclassified markers are ringed.');
 // Bind the sixteen drawn markers (not the four legend glyphs) in TikZ order; verify against the calc export.
 const drawn=[...art.querySelectorAll('path')].filter(p=>/69\.999695%|100%, 0%, 0%/.test(p.getAttribute('stroke')||'')).slice(0,16);
 if(drawn.length!==16)throw Error('Expected sixteen drawn markers');
 drawn.forEach((p,j)=>{const m=D.markers[j],t=/matrix\(([\d.]+), 0, 0, -[\d.]+, ([\d.]+),/.exec(p.getAttribute('transform')||''),nums=(p.getAttribute('d')||'').match(/-?[\d.]+/g)?.map(Number).filter((_,i)=>i%2===0)||[];
  const cx=t?Number(t[2])+Number(t[1])*(Math.min(...nums)+Math.max(...nums))/2:NaN;
  if(!(Math.abs(cx-m.x)<0.01)||(m.g===0)!==/69\.999695%/.test(p.getAttribute('stroke')))throw Error('Marker '+j+' does not match the reviewed artwork');});
 // Overlay: per-row approval shading and boundary segments, side tags, committed rings, counts.
 const overlay=svg('g',{'data-ft-overlay':'true'}),values=svg('g',{'data-mechanism-values':'true'});
 const top=D.lineY[0]-1,mid=D.splitY,bot=D.lineY[1];
 const shades=[[top,mid],[mid,bot]].map(([y0,y1])=>svg('rect',{y:y0,height:y1-y0,fill:APPROVE,'fill-opacity':.11}));
 // Two segments that coincide while the boundary is shared, so a split is visibly the same object separating.
 const lines=[[top,mid],[mid,bot]].map(([y1,y2],g)=>svg('line',{y1,y2,stroke:INK,'stroke-width':1.6,'stroke-linecap':'butt','data-ft-boundary':String(g),'data-motion-object':'true'}));
 const tagStyle={'font-size':5.6,'font-family':'sans-serif',fill:INK,y:61.6};
 const approveTag=svg('text',{...tagStyle,'text-anchor':'end',class:'ft-desktop'}),denyTag=svg('text',{...tagStyle,'text-anchor':'start',class:'ft-desktop'});
 overlay.append(...shades,...lines,approveTag,denyTag);
 const rings=svg('g',{'data-ft-rings':'true'});
 const accuracy=svg('text',{x:5.5,y:8.6,'font-size':6.6,'font-weight':'700',fill:INK,'font-family':'sans-serif','data-ft-accuracy':'true',class:'ft-desktop'});
 const counts=[41.6,66.2].map(y=>svg('text',{x:5.5,y,'font-size':6.2,fill:INK,'font-family':'sans-serif',class:'ft-desktop'}));
 values.append(rings,accuracy,...counts);
 // Narrow screens crop to the score rows; these labels replace the cropped row titles and legend.
 const mobile=svg('g',{class:'ft-mobile','aria-hidden':'true'});
 // A mask hides the clipped tails of the original row titles; the crop ends above the legend.
 mobile.append(svg('rect',{x:30,y:0,width:D.tx-30,height:66,fill:'#fff'}));
 [['A',D.rowY[0]-2.4],['B',D.rowY[1]-2.4]].forEach(([label,y])=>mobile.append(svg('text',{x:36.4,y,'font-size':8.4,'font-weight':'700',fill:INK,'font-family':'sans-serif'},label)));
 values.append(mobile);
 const narrow=window.matchMedia('(max-width:700px)');
 // The phone crop keeps the whole score span, through the ring on the last B marker.
 function layout(){art.setAttribute('viewBox',narrow.matches?'35 1 256 59':D.viewBox);if(committedOnce){paint(D.stops[committed]);commit(committed);}}
 let committedOnce=false;narrow.addEventListener('change',layout);layout();
 art.insertBefore(overlay,[...art.children].find(c=>c.localName==='path')||null);art.append(values);
 const source=document.createElement('div');source.className='ft-source';source.append(art);
 // Text ledger: the same committed counts, readable at any width and in both themes.
 const ledgerBox=document.createElement('div');ledgerBox.className='ft-ledger-box';
 const el=(tag,cls,text)=>{const e=document.createElement(tag);if(cls)e.className=cls;if(text!=null)e.textContent=text;return e;};
 const note=el('p','ft-ledger-note');
 const table=el('table','ft-ledger'),thead=el('thead'),head=el('tr'),body=el('tbody');
 ['',...GROUP].forEach(h=>{const th=el('th','',h);th.scope='col';head.append(th);});thead.append(head);table.append(thead,body);
 const rows=['Approved','Repayers approved (TPR)','Defaulters approved (FPR)','Misclassified'].map(label=>{const tr=el('tr'),th=el('th','',label);th.scope='row';tr.append(th);const cells=[0,1].map(()=>tr.appendChild(el('td')));body.append(tr);return cells;});
 const overall=el('p','ft-overall'),sweep=el('p','ft-sweep');
 // The sweep, inspectable: correct of 16 at every shared position, left (strictest) to right.
 const strip=el('div','ft-strip-box'),scap=el('p','ft-strip-note'),grid=el('div','ft-strip');
 // A grid of position/value cells: one line of 17 on desktop, wrapping to two lines on phone.
 grid.setAttribute('role','list');grid.append(el('div','ft-strip-key',null));grid.firstChild.append(el('span','ft-pos','Position'),el('span','ft-val','Correct'));grid.firstChild.setAttribute('aria-hidden','true');
 CMP.sweep.correctByPosition.forEach((c,j)=>{const cell=el('div','ft-strip-cell');cell.setAttribute('role','listitem');
  cell.setAttribute('aria-label',`Position ${j+1}: ${c} of 16 correct`);cell.append(el('span','ft-pos',String(j+1)),el('span','ft-val',String(c)));
  if(c===CMP.sweep.bestCorrect)cell.classList.add('ft-best');if(LINE_POS.includes(j+1))cell.classList.add('ft-line');grid.append(cell);});
 scap.textContent=`Correct of 16 at each of the ${CMP.sweep.cutoffs} shared positions, from strictest (1, left of every marker) to most lenient (${CMP.sweep.cutoffs}). The ${D.labels[0]}% line is position ${LINE_POS[0]}; the ${D.labels[1]}% line is position ${LINE_POS[1]} (shaded).`;
 scap.append(el('span','ft-strip-rows',' In each cell the small number is the position and the large one is the count.'));
 scap.id=NAME+'-strip-note';grid.setAttribute('aria-describedby',scap.id);strip.append(scap,grid);
 const history=el('table','ft-ledger ft-history'),hhead=el('thead'),hrow=el('tr'),hbody=el('tbody');
 ['Boundary','A errors','B errors','Accuracy'].forEach(h=>{const th=el('th','',h);th.scope='col';hrow.append(th);});hhead.append(hrow);history.append(hhead,hbody);
 const legend=el('p','ft-legend','Circles repaid; plus markers defaulted. Blue is Subgroup A, red is Subgroup B. Further left is a higher score; each row\'s boundary approves the shaded markers to its left. A dashed ring with ✗ marks a misclassification. Expand enlarges the figure.');
 ledgerBox.append(note,legend,overall,table,sweep,strip,history);
 // Visual state follows the interpolated boundary positions; nothing here writes a count.
 function paint(b,opacity=1){
  const show=b!=null,xs=show?b.map(sx):[X0,X0],split=show&&Math.abs(xs[0]-xs[1])>1e-6;
  [...shades,...lines,approveTag,denyTag].forEach(e=>{e.style.visibility=show?'visible':'hidden';e.setAttribute('opacity',opacity);});
  shades.forEach((r,g)=>{r.setAttribute('x',X0);r.setAttribute('width',Math.max(0,xs[g]-X0));});
  lines.forEach((l,g)=>{l.setAttribute('x1',xs[g]);l.setAttribute('x2',xs[g]);});
  // Tags sit at the bottom of the B segment; once the boundary splits they name B explicitly.
  // The band under row B holds no markers, only the two dashed candidate lines, so each tag
  // takes the longest wording whose measured extent clears both lines, else a bare arrow, else none.
  approveTag.setAttribute('x',xs[1]-2.2);denyTag.setAttribute('x',xs[1]+2.2);
  fit(approveTag,split?['◀ B approves','◀ B','◀']:['◀ approve','◀'],-1);fit(denyTag,split?['B denies ▶','B ▶','▶']:['deny ▶','▶'],1);
 }
 const DASH_X=D.bounds.map(sx),GAP=0.8;
 function fit(tag,words,side){
  const x=Number(tag.getAttribute('x'));
  for(const w of words){tag.textContent=w;let len=0;try{len=tag.getComputedTextLength();}catch(e){}
   // Unmeasurable (hidden on phone, or not yet in the page): keep the full wording.
   if(!(len>0))return;
   const lo=side<0?x-len:x,hi=side<0?x:x+len;
   if(!DASH_X.some(d=>d>lo-GAP&&d<hi+GAP))return;}
  tag.textContent='';
 }
 // Committed values: rings, in-figure counts, ledger tables, and data attributes.
 function commit(i){
  const L=ledgers[i],scored='errors' in L,RING=narrow.matches?{r:7,w:1.5,dash:'2.4 1.5',dy:9.6,f:8.6}:{r:6.6,w:.9,dash:'2 1.4',dy:8.4,f:6.4};committedOnce=true;
  rings.replaceChildren(...(L.wrong||[]).map(j=>{const m=D.markers[j],g=svg('g',{'data-ft-wrong':String(j)});
   // The cross sits above its ring so it never lands between adjacent markers; phone draws both heavier.
   g.append(svg('circle',{cx:m.x,cy:m.cy,r:RING.r,fill:'none',stroke:INK,'stroke-width':RING.w,'stroke-dasharray':RING.dash}),svg('text',{x:m.x,y:m.cy-RING.dy,'font-size':RING.f,'font-weight':'700',fill:INK,'font-family':'sans-serif','text-anchor':'middle'},'✗'));return g;}));
  accuracy.textContent=L.phase==='none'?'':L.phase==='decide'?'Accuracy: not yet checked':L.phase==='compare'?`Accuracy ${L.before.accuracyPct}% → ${L.accuracyPct}%`:`Accuracy ${L.correct}/16 = ${L.accuracyPct}%`;
  counts.forEach((t,g)=>{t.textContent=L.phase==='none'?'':L.phase==='decide'?`${L.approved[g]} of 8 approved`:L.phase==='compare'?`${L.before.errors[g]} → ${n(L.errors[g],'error')}`:n(L.errors[g],'error');});
  const where=L.phase==='split'?` Boundaries: Subgroup A on the ${LINE.name}, Subgroup B on its own cutoff (${L.name}).`:L.name?` Shared boundary: ${L.name}.`:'';
  note.textContent=`Step ${i+1}: toy example, the 16 drawn applicants (4 repay and 4 default per subgroup). These are not the credit-service metrics in the table above.`+where;
  const pend='–';
  [0,1].forEach(g=>{rows[0][g].textContent=L.approved?`${L.approved[g]} of 8`:pend;rows[1][g].textContent=scored?`${L.repayersApproved[g]} of 4 (${L.tpr[g]})`:pend;rows[2][g].textContent=scored?`${L.defaultersApproved[g]} of 4 (${L.fpr[g]})`:pend;rows[3][g].textContent=scored?String(L.errors[g]):pend;});
  overall.textContent=scored?`Overall: ${L.correct} of 16 correct (${L.accuracyPct}%)`+(L.phase==='split'?', with two cutoffs.':'.'):L.phase==='decide'?'Overall: decisions made; outcomes not yet checked.':'Overall: no boundary applied yet.';
  // The sweep result is established at the compare step and stays visible for the split steps.
  const S=CMP.sweep;sweep.hidden=strip.hidden=i<CMP.step;
  sweep.textContent=`All ${S.cutoffs} shared positions checked: best ${S.bestPct}% (${S.bestCorrect} of 16), reached ${S.bestCount===1?'only':'at '+S.bestCount+' positions,'} at the ${LINE.name}; every other shared position scores at most ${S.nextPct}%.`;
  hbody.replaceChildren(...ledgers.filter(H=>H.step>=2&&H.step<=i&&H.phase!=='compare').map(H=>{const tr=el('tr');[H.name,H.errors[0],H.errors[1],H.accuracyPct+'%'].forEach((v,c)=>{const cell=el(c?'td':'th','',String(v));if(!c)cell.scope='row';tr.append(cell);});
   if(L.phase==='compare'&&(H.step===FIRST.step||H.step===LINE.step))tr.className='ft-candidate';if(H.phase==='split')tr.classList.add('ft-split');return tr;}));
  history.hidden=i<2;
  root.dataset.ledger=JSON.stringify(L);root.dataset.lessonState=String(i);
 }
 let committed=0,mounted=false;
 const steps=states.map(([label,describe],i)=>({label,get description(){return committed===i?describe(ledgers[i]):'In progress: '+label+'. The rings and counts still describe the last completed step.';}}));
 return window.MLSysOriginalPlayer.create({root,outer,title:'Where do the errors go when the boundary moves?',
  scope:'Illustrative toy: the original drawing\'s 16 applicants, 8 per subgroup, each with 4 repayers (circles) and 4 defaulters (plus markers). Horizontal position is score order only: markers further left have higher scores, so a boundary approves the markers to its left and moving it right is more lenient. The dashed labels are the accuracy each candidate boundary achieves, not score cutoffs. Steps 2 to 9 use one shared boundary for both rows; steps 10 to 12 keep Subgroup A on the 81.25% line and give Subgroup B its own cutoff, the threshold adjustment discussed after the figure, which is legally restricted and shown here only to expose its trade-off. Every number is a count of drawn markers or a ratio of those counts, not the credit-service metrics in the chapter\'s table. Intermediate stops sit between drawn markers so each move crosses one marker (the first B-only move crosses two repayers). Animation speed represents nothing.',steps,
  render(i,{stage,details}){committed=i;if(!mounted){stage.append(source);details.append(ledgerBox);mounted=true;}paint(D.stops[i]);commit(i);delete root.dataset.pending;},
  transition({from,to,onFrame}){root.dataset.pending=String(to);const a=D.stops[from],b=D.stops[to];
   onFrame(p=>{if(a==null)paint(b,p);else if(b==null)paint(a,1-p);else paint([0,1].map(g=>a[g]+(b[g]-a[g])*p));root.dataset.animationProgress=String(p);});}
 });
};})();

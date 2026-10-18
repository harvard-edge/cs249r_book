/* Specialized, deterministic lesson. Animation time is reading time, not a hardware model. */
(() => {
'use strict';
let instances = 0;
window.MLSysFigureLessons = window.MLSysFigureLessons || {};
window.MLSysFigureLessons['paged-attention'] = (root, context) => {
 const prefix = 'mlsys-' + 'paged-attention' + '-' + (++instances);
 const abort = new AbortController();
 const on = (el, type, fn) => el.addEventListener(type, fn, {signal:abort.signal});
 const el = (tag, cls, text) => { const n=document.createElement(tag); if(cls)n.className=cls; if(text!==undefined)n.textContent=text; return n; };
 const box = (cls, text) => `<div class="cell ${cls}">${text}</div>`;
 root.classList.add('mlsys-v2-lesson', 'mlsys-' + 'paged-attention');
 let step=0, running=false, timer=null, destroyed=false, selectedPage=null, ownershipAnswer=null, renderedStep=-1;
 const intro=el('p','lesson-question',"How can four nonadjacent free slots hold one ordered KV sequence?"), scope=el('p','lesson-scope',"The exact eight-slot example in the figure. The added step-through follows Seq E growing one logical page at a time, then releasing its pages after the final consumer; page contents and timing are schematic.");
 const controls=el('div','lesson-controls'); controls.setAttribute('role','group'); controls.setAttribute('aria-label',intro.textContent+' playback');
 const buttons={};
 for(const [key,label] of [['play','Play'],['back','Back'],['next','Next'],['restart','Restart']]) {const b=el('button','',label);b.type='button'; b.id=prefix+'-'+key;buttons[key]=b;controls.append(b);}
 const label=el('label','','Sequence '), menu=el('select');menu.id=prefix+'-sequence';label.htmlFor=menu.id;label.append(menu);controls.append(label);
 const current=el('div','lesson-current');current.setAttribute('aria-live','polite');current.setAttribute('aria-atomic','true');
 const title=el('p','lesson-step'), narration=el('p','lesson-narration');current.append(title,narration);
 const art=el('div','lesson-art');art.setAttribute('role','group');art.setAttribute('aria-label',intro.textContent);
 const detail=el('div','lesson-detail');
 const lookup=el('div','lesson-task'), lookupLabel=el('label','','Trace Seq E logical page '), pageMenu=el('select');
 pageMenu.id=prefix+'-logical-page';lookupLabel.htmlFor=pageMenu.id;
 const placeholder=el('option','','Choose a page');placeholder.value='';pageMenu.append(placeholder);
 for(let j=0;j<4;j++){const option=el('option','',String(j));option.value=String(j);pageMenu.append(option);}
 lookupLabel.append(pageMenu);
 const lookupResult=el('p','lookup-result');lookupResult.id=prefix+'-lookup-result';lookupResult.setAttribute('aria-live','polite');pageMenu.setAttribute('aria-describedby',lookupResult.id);lookup.append(lookupLabel,lookupResult);
 const ownership=el('fieldset','lesson-task'), ownershipLegend=el('legend','','Can E’s physical pages be reassigned now?'), ownershipResult=el('p','ownership-result');
 ownershipResult.id=prefix+'-ownership-result';ownershipResult.setAttribute('aria-live','polite');ownership.setAttribute('aria-describedby',ownershipResult.id);
 const ownershipButtons={};ownership.append(ownershipLegend);
 for(const [value,text] of [['yes','Yes'],['no','No']]){const button=el('button','',text);button.type='button';button.id=prefix+'-reassign-'+value;ownershipButtons[value]=button;ownership.append(button);}
 ownership.append(ownershipResult);
 root.append(intro,scope,controls,current,lookup,art,ownership,detail);


 const stages=[
 ['(a) Reserved but unused','Seq A uses two of four reserved slots and Seq B uses three of four. Hatched slots belong to those reservations and cannot serve another request: internal fragmentation.'],
 ['(b) Free, but not contiguous','In the separate after-completion layout, C occupies slots 2–3 and D occupies 6–7. Four slots are free, but the largest gap is two. A four-slot contiguous reservation for E cannot fit.'],
 ['(c) Allocate E page 0','E’s first logical KV page maps to physical slot 0. The block table records ownership; C and D keep their existing slots.'],
 ['Grow E: page 1','The next logical page maps to physical slot 1. Only allocated pages occupy physical storage; future logical pages have no table entry yet.'],
 ['Grow E across the gap','E page 2 maps to physical slot 4, skipping C’s slots. Logical order stays 0, 1, 2 even though physical slots are 0, 1, 4.'],
 ['The original paged result','E page 3 maps to physical slot 5. E now occupies 0, 1, 4, 5, and the unfilled tail of page 3 remains unused. C and D have not moved.'],
 ['E finishes; last use pending','E has stopped generating, but its last KV consumer has not finished. Its table and all four physical pages remain allocated until that consumer releases them.'],
 ['Release E’s pages','After the last consumer finishes, E’s table entries are removed and physical slots 0, 1, 4, 5 return to the free pool. C and D remain resident.']
 ];
 const map=[0,1,4,5];
 const slots=(items)=>`<div class="slots">${items.map(x=>box(x[1],x[0])).join('')}</div>`;
 function state(s){return {count:s<2?0:s<5?s-1:s===7?0:4,finalConsumerFinished:s===7};}
 function draw(s){
  const {count,finalConsumerFinished}=state(s);
  if(selectedPage!==null&&selectedPage>=count)selectedPage=null;
  const tracing=selectedPage!==null&&selectedPage<count;
  pageMenu.disabled=count===0;
  Array.from(pageMenu.options).slice(1).forEach((option,j)=>{option.disabled=j>=count;});
  pageMenu.value=tracing?String(selectedPage):'';
  lookupResult.textContent=tracing?`E logical page ${selectedPage} → block-table entry ${selectedPage} → physical slot ${map[selectedPage]}. ${selectedPage>=2?'Slots 2–3 still belong to C; logical and physical indices are not interchangeable.':'The block table, rather than the logical index alone, determines the physical slot.'}`:count?'Choose an allocated logical page to trace its block-table entry and physical slot. Unallocated pages are unavailable.':s===7?'E’s table has been removed. Its former pages cannot be looked up.':'Allocate E’s first page in step 3 to enable lookup.';
  ownership.disabled=s<6;
  for(const [answer,button] of Object.entries(ownershipButtons))button.setAttribute('aria-pressed',String(ownershipAnswer===answer));
  ownershipResult.textContent=s<6?'At steps 7–8, decide whether E’s final consumer still needs these pages.':ownershipAnswer===null?(finalConsumerFinished?'The final consumer has finished. E’s table is removed and its four slots are free.':'E has stopped generating. Its final consumer is still reading the retained KV state.'):`${(ownershipAnswer==='yes')===finalConsumerFinished?'Correct.':'Follow the final consumer.'} ${finalConsumerFinished?'Yes: after the last consumer finishes, slots '+map.join(', ')+' can be reassigned. C and D keep their pages.':'No: stopping generation is insufficient. The last consumer still needs all four pages, so reassigning one would overwrite live KV state.'}`;

  const physical=Array.from({length:8},(_,i)=>i===2||i===3?['C','other']:i===6||i===7?['D','other']:['free','empty']);
  for(let j=0;j<count;j++)physical[map[j]]=['E'+j,(tracing&&selectedPage===j?'lookup-selected ':'')+'compute '+(j===3?'tail-end':'')+(s===j+2?' active':'')];
  const logical=map.map((_,j)=>[String(j),(tracing&&selectedPage===j?'lookup-selected ':'')+(j<count?'compute '+(j===3?'tail-end':''):'empty')]);
  const table=map.map((v,j)=>`<div class="mapping-row ${tracing&&selectedPage===j?'lookup-selected':''}"><span>page ${j}</span><strong>${j<count?'slot '+v:'—'}</strong></div>`).join('');
  art.innerHTML=`<div class="fragment-panels"><section class="fragment-panel"><p class="panel-title">(a) Contiguous reservation</p>${slots([['A','compute'],['A','compute'],['','compute tail'],['','compute tail'],['B','compute'],['B','compute'],['B','compute'],['','compute tail']])}<p class="panel-note">Seq A: 2 of 4 used · Seq B: 3 of 4 used</p><p class="panel-note">Each sequence reserves the maximum length of 4 slots.</p></section><section class="fragment-panel"><p class="panel-title">(b) After sequences finish</p>${slots([['','empty'],['','empty'],['C','other'],['C','other'],['','empty'],['','empty'],['D','other'],['D','other']])}<p class="need">Seq E needs 4 contiguous slots</p><p class="panel-note">4 slots free, but the largest gap is 2.</p></section></div><section class="paged-panel"><p class="panel-title">(c) PagedAttention: ${s===7?'Seq E released':count===4?'the same memory holds Seq E':'allocate Seq E on demand'}</p><div class="mapping"><div><p class="panel-note">Seq E logical pages</p>${slots(logical)}</div><span class="map-arrow">→</span><div><p class="panel-note">block table</p><div class="block-table coord">${table}</div></div><span class="map-arrow">→</span><div><p class="panel-note">physical slots</p>${slots(physical)}<div class="slot-indices">${Array.from({length:8},(_,i)=>`<span>${i}</span>`).join('')}</div></div></div></section><div class="legend"><span class="compute">KV entries in use</span><span class="tail">reserved, unused tail</span><span class="empty">free</span><span class="other">other sequences</span></div>`;
  detail.innerHTML=`<p><strong>Seq E’s block table:</strong> ${count?map.slice(0,count).map((v,j)=>j+' → '+v).join(', '):'no allocated pages'}.</p><p><strong>Physical ownership:</strong> E holds ${count} of 8 slots; C and D hold 4; ${4-count} are free.${s===6?' E’s pages are retained for the final reader.':''}</p><p><strong>Capacity stays fixed.</strong> Paging removes the need for one contiguous run; it does not create HBM, eliminate final-page slack, or make an arbitrarily long resident sequence fit. Panels (a) and (b) are distinct layouts from the source example, not a claim that A becomes C.</p>`;
 }
 stages.forEach((s,i)=>{const o=el('option','',`${i+1}. ${s[0]}`);o.value=i;menu.append(o);});
 function render(){
   if(renderedStep!==step){ownershipAnswer=null;renderedStep=step;}
   root.dataset.lessonStep=String(step);root.dataset.lessonRunning=String(running);
   title.textContent=`${step+1} / ${stages.length} · ${stages[step][0]}`;narration.textContent=stages[step][1];menu.value=String(step);
   buttons.play.textContent=running?'Pause':step===stages.length-1?'Replay':'Play';buttons.play.setAttribute('aria-pressed',String(running));
   buttons.back.disabled=step===0;buttons.next.disabled=step===stages.length-1;
   draw(step);
 }
 function pause(){running=false;clearTimeout(timer);timer=null;if(!destroyed)render();}
 function schedule(){timer=setTimeout(()=>{if(!running||destroyed)return;if(document.hidden||root.closest('[hidden]')){pause();return;}if(step<stages.length-1)step++;if(step===stages.length-1)running=false;render();if(running)schedule();},Math.max(5500,stages[step][1].split(/\s+/).length*260));}
 on(buttons.play,'click',()=>{if(running){pause();return;}if(step===stages.length-1)step=0;running=true;render();schedule();});
 on(buttons.back,'click',()=>{pause();step=Math.max(0,step-1);render();});
 on(buttons.next,'click',()=>{pause();step=Math.min(stages.length-1,step+1);render();});
 on(buttons.restart,'click',()=>{pause();step=0;render();});
 on(menu,'change',()=>{const selected=Number(menu.value);pause();step=selected;render();});
 on(pageMenu,'change',()=>{const value=pageMenu.value;pause();selectedPage=value===''?null:Number(value);render();});
 for(const [answer,button] of Object.entries(ownershipButtons))on(button,'click',()=>{pause();ownershipAnswer=answer;render();});
 on(document,'visibilitychange',()=>{if(document.hidden)pause();});on(window,'beforeprint',pause);
 on(root,'mlsys-view-change',e=>{if(e.detail?.view!=='explore')pause();});
 const intersection=typeof IntersectionObserver==='function'?new IntersectionObserver(items=>{if(items.some(x=>!x.isIntersecting)&&running)pause();}):null;intersection?.observe(root);
 const hidden=new MutationObserver(()=>{if(running&&root.closest('[hidden]'))pause();});hidden.observe(document.documentElement,{attributes:true,attributeFilter:['hidden'],subtree:true});
 const motion=window.matchMedia('(prefers-reduced-motion: reduce)');on(motion,'change',()=>{if(motion.matches)pause();});
 render();
 return {pause,destroy(){pause();destroyed=true;abort.abort();intersection?.disconnect();hidden.disconnect();root.replaceChildren();}};
};
})();

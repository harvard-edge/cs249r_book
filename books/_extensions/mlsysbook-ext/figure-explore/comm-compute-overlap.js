/* Specialized, deterministic lesson. Animation time is reading time, not a hardware model. */
(() => {
'use strict';
let instances = 0;
window.MLSysFigureLessons = window.MLSysFigureLessons || {};
window.MLSysFigureLessons['comm-compute-overlap'] = (root, context) => {
 const prefix = 'mlsys-' + 'comm-compute-overlap' + '-' + (++instances);
 const abort = new AbortController();
 const on = (el, type, fn) => el.addEventListener(type, fn, {signal:abort.signal});
 const el = (tag, cls, text) => { const n=document.createElement(tag); if(cls)n.className=cls; if(text!==undefined)n.textContent=text; return n; };
 const box = (cls, text) => `<div class="cell ${cls}">${text}</div>`;
 root.classList.add('mlsys-v2-lesson', 'mlsys-' + 'comm-compute-overlap');
 let step=0, running=false, timer=null, destroyed=false;
 const intro=el('p','lesson-question','Which gradient is ready to communicate, and what must the optimizer wait for?'), scope=el('p','lesson-scope','The original four-layer schematic. Durations and playback intervals are illustrative; each amber o slice occupies the GPU and cannot overlap its next compute block.');
 const controls=el('div','lesson-controls'); controls.setAttribute('role','group'); controls.setAttribute('aria-label',intro.textContent+' playback');
 const buttons={};
 for(const [key,label] of [['play','Play'],['back','Back'],['next','Next'],['restart','Restart']]) {const b=el('button','',label);b.type='button'; b.id=prefix+'-'+key;buttons[key]=b;controls.append(b);}
 const label=el('label','','Sequence '), menu=el('select');menu.id=prefix+'-sequence';label.htmlFor=menu.id;label.append(menu);controls.append(label);
 const current=el('div','lesson-current');current.setAttribute('aria-live','polite');current.setAttribute('aria-atomic','true');
 const title=el('p','lesson-step'), narration=el('p','lesson-narration');current.append(title,narration);
 const art=el('div','lesson-art');art.setAttribute('role','group');art.setAttribute('aria-label','Which gradient is ready to communicate, and what must the optimizer wait for?');
 const detail=el('div','lesson-detail');
 root.append(intro,scope,controls,current,art,detail);

 const stages=[
 ['Backward L4','Only L4 is computing. Its gradient is not ready for AllReduce until backward L4 finishes.'],
 ['L4 ready; overhead o','L4’s gradient is ready, but launch overhead occupies the GPU. Backward L3 has not started.'],
 ['L3 compute + L4 transfer','AllReduce L4 runs on the network while the GPU computes backward L3. L3’s gradient is not ready yet.'],
 ['L3 ready; overhead o','L4’s reduction is complete in this schematic. The GPU performs L3’s nonoverlappable launch overhead before starting L2.'],
 ['L2 compute + L3 transfer','Backward L2 and AllReduce L3 proceed concurrently. Their work remains distinct even though their elapsed-time windows overlap.'],
 ['L2 ready; overhead o','L3’s reduction is complete. The GPU handles L2’s launch overhead; L1 compute has not started.'],
 ['L1 compute + L2 transfer','The final backward layer computes while AllReduce L2 runs. No earlier-layer compute will remain after L1.'],
 ['L1 ready; overhead o','L1’s gradient is ready. Its launch overhead and following transfer lie on the exposed tail.'],
 ['GPU waits for L1','AllReduce L1 runs with no backward compute to hide it. The optimizer cannot use a gradient whose collective is still in flight.'],
 ['Optimizer dependency satisfied','All four reductions are complete. The optimizer can now consume the globally reduced gradients.'],
 ['Step’s optimizer completes','Three transfers were hidden by subsequent backward work. L1’s transfer and all nonoverlappable overhead remain paid; no measured speedup is claimed.']
 ];
 const activeCompute=[0,-1,1,-1,2,-1,3,-1,-1,4,-1];
 function draw(s){
  const col=[1,3,5,7], layer=[4,3,2,1];
  const compute=layer.map((l,i)=>`<div style="grid-column:${col[i]};grid-row:1" class="cell compute ${s>2*i?'done':s===2*i?'active':'future'}">Backward L${l}</div><div style="grid-column:${col[i]+1};grid-row:1" class="overhead ${s===2*i+1?'active':s<2*i+1?'future':''}" aria-label="L${l} nonoverlappable GPU overhead">o</div>`).join('');
  const network=layer.map((l,i)=>`<div style="grid-column:${col[i]+2};grid-row:2" class="cell transfer ${s>=2*i+3?'done':s===2*i+2?'active':'future'}">AllReduce L${l}</div>`).join('');
  art.innerHTML=`<div class="legend"><span>Outline: current state</span><span>✓: completed</span><span>Dashed: scheduled region or wait</span></div><p class="scroll-hint">Scroll the timeline horizontally →. The current state is also listed below.</p><div class="scroll-diagram" tabindex="0" aria-label="Two-lane timeline; scroll horizontally on a narrow screen"><div class="overlap-layout timeline"><div class="lane-name">Compute</div><div class="overlap-grid">${compute}<div style="grid-column:9;grid-row:1" class="cell ${s===8?'active':''} future">GPU idle</div><div style="grid-column:10;grid-row:1" class="cell compute ${s===9?'active':s===10?'done':'future'}">Optimizer</div>${network}<div class="hidden-bracket">hidden behind backward compute</div><div class="exposed-bracket">exposed tail</div><div class="time-axis">time →</div></div><div class="lane-name network-label">Network</div></div></div>`;
  const rows=layer.map((l,i)=>{const ready=s>=2*i+1;const reduced=s>=2*i+3;return `<tr><th>L${l}</th><td>${ready?'ready':'not ready'}</td><td>${reduced?'complete':s===2*i+2?'in flight':'not launched'}</td></tr>`;}).join('');
  detail.innerHTML=`<p><strong>GPU now:</strong> ${s%2===1&&s<8?'launch overhead o':s===8?'idle, waiting for L1':s===9?'optimizer':s===10?'step complete':'backward L'+(4-activeCompute[s])}.</p><table class="state-table"><caption>Gradient dependencies at this step</caption><thead><tr><th>Layer</th><th>Local gradient</th><th>AllReduce</th></tr></thead><tbody>${rows}</tbody></table><p>In this schematic each transfer finishes within the next backward block. Actual queueing or slower transfers can expose additional wait. Every o slice occupies the GPU.</p>`;
 }

 stages.forEach((s,i)=>{const o=el('option','',`${i+1}. ${s[0]}`);o.value=i;menu.append(o);});
 function render(){
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
 on(document,'visibilitychange',()=>{if(document.hidden)pause();});on(window,'beforeprint',pause);
 on(root,'mlsys-view-change',e=>{if(e.detail?.view!=='explore')pause();});
 const intersection=typeof IntersectionObserver==='function'?new IntersectionObserver(items=>{if(items.some(x=>!x.isIntersecting)&&running)pause();}):null;intersection?.observe(root);
 const hidden=new MutationObserver(()=>{if(running&&root.closest('[hidden]'))pause();});hidden.observe(document.documentElement,{attributes:true,attributeFilter:['hidden'],subtree:true});
 const motion=window.matchMedia('(prefers-reduced-motion: reduce)');on(motion,'change',()=>{if(motion.matches)pause();});
 render();
 return {pause,destroy(){pause();destroyed=true;abort.abort();intersection?.disconnect();hidden.disconnect();root.replaceChildren();}};
};
})();

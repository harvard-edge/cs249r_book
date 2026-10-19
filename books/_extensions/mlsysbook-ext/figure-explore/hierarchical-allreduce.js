/* Specialized, deterministic lesson. Animation time is reading time, not a hardware model. */
(() => {
'use strict';
let instances = 0;
window.MLSysFigureLessons = window.MLSysFigureLessons || {};
window.MLSysFigureLessons['hierarchical-allreduce'] = (root, context) => {
 const prefix = 'mlsys-' + 'hierarchical-allreduce' + '-' + (++instances);
 const abort = new AbortController();
 const on = (el, type, fn) => el.addEventListener(type, fn, {signal:abort.signal});
 const el = (tag, cls, text) => { const n=document.createElement(tag); if(cls)n.className=cls; if(text!==undefined)n.textContent=text; return n; };
 const box = (cls, text) => `<div class="cell ${cls}">${text}</div>`;
 root.classList.add('mlsys-v2-lesson', 'mlsys-' + 'hierarchical-allreduce');
 let step=0, running=false, timer=null, destroyed=false;
 const intro=el('p','lesson-question','How do node-local partial sums become a full result on every GPU?'), scope=el('p','lesson-scope','The figure’s two nodes and two representative GPUs per node, with symbolic shards A and B. Subscripts identify contributors; playback does not encode bandwidth or elapsed time.');
 const controls=el('div','lesson-controls'); controls.setAttribute('role','group'); controls.setAttribute('aria-label',intro.textContent+' playback');
 const buttons={};
 for(const [key,label] of [['play','Play'],['back','Back'],['next','Next'],['restart','Restart']]) {const b=el('button','',label);b.type='button'; b.id=prefix+'-'+key;buttons[key]=b;controls.append(b);}
 const label=el('label','','Sequence '), menu=el('select');menu.id=prefix+'-sequence';label.htmlFor=menu.id;label.append(menu);controls.append(label);
 const current=el('div','lesson-current');current.setAttribute('aria-live','polite');current.setAttribute('aria-atomic','true');
 const title=el('p','lesson-step'), narration=el('p','lesson-narration');current.append(title,narration);
 const art=el('div','lesson-art');art.setAttribute('role','group');art.setAttribute('aria-label','How do node-local partial sums become a full result on every GPU?');
 const detail=el('div','lesson-detail');
 const lower=el('div','lesson-mobile-controls');
 const previous=el('button','','Previous stage'),following=el('button','','Next stage');
 previous.type=following.type='button';previous.id=prefix+'-lower-back';following.id=prefix+'-lower-next';lower.append(previous,following);
 root.append(intro,scope,controls,current,art,lower,detail);

 const stages=[
 ['Per-rank contributions','Every GPU starts with contributions to both shards. A₀₀ means Node 0, GPU 0’s contribution to shard A. No sum is global yet.'],
 ['1. Local ReduceScatter','NVLink reduces within each node and scatters ownership: GPU 0 holds the node’s A partial sum; GPU 1 holds its B partial sum. The other shard is not held there.'],
 ['2. Exchange corresponding shards','Only corresponding GPUs communicate across nodes: A partial sums on the left InfiniBand link, B partial sums on the right. The node partial sums are not full sums yet.'],
 ['2. Shard AllReduce completes','The A reduction returns ΣA to GPU 0 of both nodes; the B reduction returns ΣB to GPU 1. Each GPU owns one globally reduced shard.'],
 ['3. Local AllGather','Each node exchanges its two globally reduced shards over NVLink. This copies the completed sums to the other GPU; it does not reduce them a second time.'],
 ['Full result on every GPU','All four GPUs now hold [ΣA, ΣB]. Inter-node work was partitioned by shard; the final local distribution restored the full result on every rank.']
 ];
 function node(n,phase,s){
   const started=phase===0||s>=phase*2;
   let contents=[];
   for(let g=0;g<2;g++){
    let shards;
    if(!started) shards=[box('empty','—'),box('empty','—')];
    else if(phase===0&&s===0) shards=[box('compute',`A<sub>${n}${g}</sub>`),box('compute',`B<sub>${n}${g}</sub>`)];
    else {
     const global=phase===2||phase===1&&s>=3;
     const all=phase===2&&s>=5;
     shards=['A','B'].map((letter,j)=>{
      if(j===g||all)return box(global?'storage':'partial',global?`Σ${letter}`:`${letter}<sub>${n}</sub>`);
      if(phase===2&&s===4)return box('transfer',`← Σ${letter}`);
      return box('empty','not held');
     });
    }
    contents.push(`<div class="gpu compute"><strong>GPU ${g}</strong><div class="shards">${shards.join('')}</div></div>`);
   }
   return `<div class="node"><strong>Node ${n}</strong><div class="gpu-pair">${contents[0]}<span class="nvlink ${started&&phase!==1?'lit':''}" aria-label="NVLink">↔</span>${contents[1]}</div></div>`;
 }
 function draw(s){
  const names=['1. ReduceScatter','2. Shard AllReduce','3. AllGather'];
  art.innerHTML=`<div class="phases">${names.map((name,p)=>`<section class="phase ${Math.min(2,Math.floor(s/2))===p?'phase-current':''}"><p class="phase-title">${name}</p><p class="phase-sub">${p===1?'InfiniBand, across nodes':'NVLink, within each node'}</p>${node(0,p,s)}<div class="fabric ${p===1&&s>=2?'fabric-live':''}"><span><em>${p===1&&s===2?'A₀ ↕ A₁':p===1&&s>=3?'ΣA ↕ ΣA':''}</em></span><span><em>${p===1&&s===2?'B₀ ↕ B₁':p===1&&s>=3?'ΣB ↕ ΣB':''}</em></span></div>${node(1,p,s)}<p class="phase-sub">${p===0?(s===0?'per-rank contributions':'node partial sums'):p===1?'one shard per GPU: M / Gnode':'local distribution'}</p></section>`).join('')}</div><div class="legend"><span class="partial">Aₙ / Bₙ: node partial sum</span><span class="storage">ΣA / ΣB: full sum</span><span>Dashed: shard not held</span></div>`;
  detail.innerHTML='<p><strong>Reduction identity:</strong> A₀ = A₀₀ + A₀₁; A₁ = A₁₀ + A₁₁; ΣA = A₀ + A₁. The same relation holds for B.</p><p><strong>Ownership:</strong> ReduceScatter partitions local ownership; shard AllReduce combines nodes; AllGather copies the full result to each local rank. The panels preserve completed earlier phases.</p>';
 }

 stages.forEach((s,i)=>{const o=el('option','',`${i+1}. ${s[0]}`);o.value=i;menu.append(o);});
 function render(){
   root.dataset.lessonStep=String(step);root.dataset.lessonRunning=String(running);
   title.textContent=`${step+1} / ${stages.length} · ${stages[step][0]}`;narration.textContent=stages[step][1];menu.value=String(step);
   buttons.play.textContent=running?'Pause':step===stages.length-1?'Replay':'Play';buttons.play.setAttribute('aria-pressed',String(running));
   buttons.back.disabled=step===0;buttons.next.disabled=step===stages.length-1;
   previous.disabled=step===0;following.disabled=step===stages.length-1;draw(step);
 }
 function pause(){running=false;clearTimeout(timer);timer=null;if(!destroyed)render();}
 function schedule(){timer=setTimeout(()=>{if(!running||destroyed)return;if(document.hidden||root.closest('[hidden]')){pause();return;}if(step<stages.length-1)step++;if(step===stages.length-1)running=false;render();if(running)schedule();},Math.max(5500,stages[step][1].split(/\s+/).length*260));}
 on(buttons.play,'click',()=>{if(running){pause();return;}if(step===stages.length-1)step=0;running=true;render();schedule();});
 on(previous,'click',()=>{pause();step=Math.max(0,step-1);render();});
 on(following,'click',()=>{pause();step=Math.min(stages.length-1,step+1);render();});
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

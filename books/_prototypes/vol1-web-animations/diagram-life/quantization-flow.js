/* Diagram-first review prototype. Durations illustrate logical flow, not hardware timing. */
(() => {
  window.MLSysFigureLessons = window.MLSysFigureLessons || {};
  window.MLSysFigureLessons['quantized-matmul'] = root => {
    const abort = new AbortController(), on = (n,e,f) => n.addEventListener(e,f,{signal:abort.signal});
    const el=(t,c,s)=>{const n=document.createElement(t);if(c)n.className=c;if(s!==undefined)n.textContent=s;return n;};
    const ns='http://www.w3.org/2000/svg', svgEl=(t,a={},s)=>{const n=document.createElementNS(ns,t);for(const[k,v]of Object.entries(a))n.setAttribute(k,v);if(s!==undefined)n.textContent=s;return n;};
    const x=[.31,-.52,.77],w=[.42,-.18,.24],round=n=>Math.sign(n)*Math.floor(Math.abs(n)+.5),fmt=n=>String(+n.toFixed(6));
    let scale=.1,step=0,running=false,animations=[],timer=null,moving=false,generation=0,dead=false;
    const reduced=matchMedia('(prefers-reduced-motion: reduce)');
    root.classList.add('mlsys-flow');
    const intro=el('p','flow-intro','Watch values move through the quantized pipeline. The red register keeps its sum as each product arrives.');
    const controls=el('div','flow-controls'),buttons={};
    for(const[k,label]of [['play','Play'],['back','Back'],['next','Next'],['restart','Restart']]){const b=el('button','',label);b.type='button';buttons[k]=b;controls.append(b);}
    const label=el('label','','Sequence '),menu=el('select');label.append(menu);controls.append(label);
    const stages=['Start with float inputs','Quantize activations','Quantize weights','Accumulate pair 1','Accumulate pair 2','Accumulate pair 3','Requantize the sum','Activate and write output'];
    stages.forEach((s,i)=>{const o=el('option','',`${i+1}. ${s}`);o.value=i;menu.append(o);});
    const narration=el('p','flow-narration');narration.setAttribute('aria-live','polite');
    const viewport=el('div','flow-viewport');viewport.tabIndex=0;viewport.setAttribute('role','region');viewport.setAttribute('aria-label','Quantized pipeline; scroll to inspect the full diagram');
    const svg=svgEl('svg',{viewBox:'0 0 1110 380',role:'img','aria-label':'Two float inputs pass through violet quantizers, blue multiplication, a red INT32 accumulator, green requantization, activation, and INT8 output.'});
    const defs=svgEl('defs'),marker=svgEl('marker',{id:root.id+'-flow-arrow',viewBox:'0 0 10 10',refX:9,refY:5,markerWidth:7,markerHeight:7,orient:'auto-start-reverse'});marker.append(svgEl('path',{d:'M0 0L10 5L0 10Z',fill:'currentColor'}));defs.append(marker);svg.append(defs);
    const links=svgEl('g',{class:'flow-links'}),nodes=svgEl('g'),tokens=svgEl('g');svg.append(links,nodes,tokens);
    const arrow=d=>links.append(svgEl('path',{d,fill:'none','marker-end':`url(#${marker.id})`}));
    arrow('M120 90H165');arrow('M120 290H165');arrow('M295 90H325V190H355');arrow('M295 290H325V190H355');
    arrow('M475 190H515');arrow('M635 190H675');arrow('M805 190H845');arrow('M945 190H985');
    const blocks={};
    function block(key,x,y,width,title,kind){const g=svgEl('g',{class:`flow-block flow-${kind}`});g.append(svgEl('rect',{x,y,width,height:86,rx:5}));const titleText=svgEl('text',{x:x+width/2,y:y+26,class:'flow-title','text-anchor':'middle'},title);const value=svgEl('text',{x:x+width/2,y:y+57,class:'flow-value','text-anchor':'middle'},'—');g.append(titleText,value);nodes.append(g);blocks[key]={g,value,titleText,original:[x,y,width],x:x+width/2,y:y+43};}
    block('x',10,47,110,'Activation x','input');block('w',10,247,110,'Weight w','input');
    block('qx',165,47,130,'Quantize x','quant');block('qw',165,247,130,'Quantize w','quant');
    block('mul',355,147,120,'Multiply','compute');block('acc',515,147,120,'INT32 sum','acc');
    block('req',675,147,130,'Requantize','output');block('act',845,147,100,'ReLU','activation');block('out',985,147,115,'INT8 output','output');
    const registerNote=svgEl('text',{x:575,y:268,class:'flow-note','text-anchor':'middle'},'Retained register');nodes.append(registerNote);
    nodes.append(svgEl('text',{x:415,y:268,class:'flow-note','text-anchor':'middle'},'Zero-points removed'),svgEl('text',{x:740,y:268,class:'flow-note','text-anchor':'middle'},'Scale + round'));
    const identity=svgEl('text',{x:555,y:350,class:'flow-note','text-anchor':'middle'},'Illustrative dataflow • one 1 × 3 by 3 × 1 dot product • no bias');nodes.append(identity);
    viewport.append(svg);
    const pan=el('p','flow-pan','On a narrow screen, the same pipeline stacks vertically. Scroll inside the diagram to inspect it; playback follows the active block.');
    const calculation=el('details','flow-calculation'),summary=el('summary','','Inspect this step’s calculation'),equation=el('p','flow-equation'),scaleLabel=el('label','','Activation scale '),scaleMenu=el('select');
    for(const[v,s]of [['0.1','0.10 — coarser'],['0.05','0.05 — finer']]){const o=el('option','',s);o.value=v;scaleMenu.append(o);}scaleLabel.append(scaleMenu);
    const assumptions=el('p','flow-assumptions','Full inputs: x = [0.31, −0.52, 0.77]; w = [0.42, −0.18, 0.24]. Signed INT8; nearest rounding with ties away from zero, then clamping. Zero-points: x = 3, w = 0, output = −4. Weight scale = 0.05; output scale = 0.02. ReLU’s stored zero is −4. These inputs do not clip. Finer spacing narrows the representable range.');
    calculation.append(summary,equation,scaleLabel,assumptions);
    const scope=el('p','flow-scope','Worked case: one 1 × 3 by 3 × 1 dot product, no bias, with ReLU. Value movement shows logical dependencies, not measured memory transfers or execution time. The full input arrays remain available; each pair is read once into this reduction.');
    root.append(intro,controls,narration,viewport,pan,calculation,scope);
    function data(){const qx=x.map(v=>round(v/scale)+3),qw=w.map(v=>round(v/.05)),terms=qx.map((v,i)=>(v-3)*qw[i]),sum=terms.reduce((a,b)=>a+b,0),out=round(sum*scale*.05/.02)-4;return{qx,qw,terms,sum,out,real:(out+4)*.02};}
    const narrow=()=>matchMedia('(max-width:600px)').matches;
    function layout(){const phone=narrow(),positions={x:[10,10,140],w:[180,10,140],qx:[10,130,140],qw:[180,130,140],mul:[100,280,130],acc:[100,410,130],req:[100,540,130],act:[10,670,135],out:[185,670,135]};svg.setAttribute('viewBox',phone?'0 0 330 790':'0 0 1110 380');for(const[k,b]of Object.entries(blocks)){const[x,y,width]=phone?positions[k]:b.original;Object.entries({x,y,width}).forEach(([a,v])=>b.g.firstChild.setAttribute(a,v));b.x=x+width/2;b.y=y+43;b.titleText.setAttribute('x',b.x);b.titleText.setAttribute('y',y+26);b.value.setAttribute('x',b.x);b.value.setAttribute('y',y+57);}const paths=phone?['M80 96V130','M250 96V130','M80 216V245H165V280','M250 216V245H165V280','M165 366V410','M165 496V540','M165 626V650H77.5V670','M145 713H185']:['M120 90H165','M120 290H165','M295 90H325V190H355','M295 290H325V190H355','M475 190H515','M635 190H675','M805 190H845','M945 190H985'];[...links.children].forEach((p,i)=>p.setAttribute('d',paths[i]));nodes.querySelectorAll('.flow-note').forEach(n=>n.style.display=phone?'none':'');}
    const count=()=>Math.max(0,Math.min(3,step-2));
    function render(settled=true){layout();const d=data(),n=count(),sum=d.terms.slice(0,n).reduce((a,b)=>a+b,0),prev=d.terms.slice(0,Math.max(0,n-1)).reduce((a,b)=>a+b,0);
      root.dataset.stage=step;root.dataset.accumulator=settled?sum:(step>=3&&step<=5?prev:sum);root.dataset.finalAccumulator=d.sum;root.dataset.output=d.out;root.dataset.reconstructed=fmt(d.real);root.dataset.moving=String(!settled);
      menu.value=step;scaleMenu.value=scale;buttons.back.disabled=step===0;buttons.next.disabled=step===7;buttons.play.textContent=running?'Pause':step===7?'Replay':'Play';buttons.play.setAttribute('aria-pressed',String(running));
      const read=Math.max(0,n-1);blocks.x.value.textContent=n?fmt(x[read]):'3 × FP32';blocks.w.value.textContent=n?fmt(w[read]):'3 × FP32';
      blocks.qx.value.textContent=step>=1?(settled||step>1?d.qx.join(', '):'converting…'):'—';
      blocks.qw.value.textContent=step>=2?(settled||step>2?d.qw.join(', '):'converting…'):'—';
      blocks.mul.value.textContent=n?(settled||step>5?String(d.terms[n-1]):'…'):'—';blocks.acc.value.textContent=String(settled?sum:step>=3&&step<=5?prev:sum);
      blocks.req.value.textContent=step>=6&&(settled||step>6)?String(d.out):'—';blocks.act.value.textContent=step===7&&settled?String(d.out):'—';blocks.out.value.textContent=step===7&&settled?String(d.out):'—';
      const active=step===0?['x','w']:step===1?['qx']:step===2?['qw']:step<=5?['mul','acc']:step===6?['req']:['act','out'];for(const[k,b]of Object.entries(blocks))b.g.classList.toggle('is-active',active.includes(k));
      const descriptions=[`Float inputs wait at the two branches; the accumulator starts at zero.`,`Activation values pass through their quantizer and become INT8 values ${d.qx.join(', ')}.`,`Weights pass through their quantizer and become INT8 values ${d.qw.join(', ')}.`,`Pair 1 contributes ${d.terms[0]}. The register ${settled?'now holds':'keeps its previous sum until arrival:'} ${settled?d.terms[0]:0}.`,`Pair 2 contributes ${d.terms[1]}; the first contribution stays in the register. ${prev} + ${d.terms[1]} = ${sum}.`,`Pair 3 contributes ${d.terms[2]}. ${prev} + ${d.terms[2]} = ${sum}; only now is the reduction complete.`,`The completed INT32 sum ${d.sum} enters requantization; scaling and rounding produce INT8 value ${d.out}.`,`ReLU preserves this positive value. Output integer ${d.out} represents ${fmt(d.real)}, compared with the float reference 0.4086.`];
      narration.textContent=`${step+1} / 8 · ${descriptions[step]}`;
      equation.textContent=step===0?'0.31 × 0.42 + (−0.52) × (−0.18) + 0.77 × 0.24 = 0.4086':step===1?`qₓ = round(x / ${scale}) + 3 → [${d.qx.join(', ')}]`:step===2?`q𝓌 = round(w / 0.05) → [${d.qw.join(', ')}]`:step<=5?`Pair ${n}: (${d.qx[n-1]} − 3) × (${d.qw[n-1]} − 0) = ${d.terms[n-1]}. Running sum: ${prev} + ${d.terms[n-1]} = ${sum}.`:step===6?`round(${d.sum} × ${fmt(scale*.05/.02)}) − 4 = ${d.out}`:`max(−4, ${d.out}) = ${d.out}; (${d.out} + 4) × 0.02 = ${fmt(d.real)}. Absolute error ${fmt(Math.abs(d.real-.4086))}. Both scales store six INT8 operand bytes; the INT32 register uses four bytes, excluding metadata and workspace.`;
      if(!settled){
        if(step===1||step===2)narration.textContent=`${step+1} / 8 · Three ${step===1?'activation':'weight'} values pass through their quantizer in order. Each arrival adds one converted INT8 value.`;
        else if(step>=3&&step<=5){narration.textContent=`${step+1} / 8 · Read pair ${n}, remove zero-points, and multiply. The register retains ${prev} until product ${d.terms[n-1]} arrives.`;equation.textContent=`Pending pair ${n}: (${d.qx[n-1]} − 3) × (${d.qw[n-1]} − 0) = ${d.terms[n-1]}. Register still holds ${prev}; after arrival, ${prev} + ${d.terms[n-1]} = ${sum}.`;}
        else if(step===6)narration.textContent=`7 / 8 · Read the completed sum ${d.sum} from the retained register. Requantization will scale and round it into output integer ${d.out}.`;
        else if(step===7)narration.textContent=`8 / 8 · Read integer ${d.out} from requantization and pass it through ReLU. Write the final output after arrival.`;
      }
    }
    function focusBlock(){if(narrow()){const ys=[65,130,130,300,300,300,565,690];viewport.scrollTo({left:0,top:Math.max(0,ys[step]/790*svg.getBoundingClientRect().height-viewport.clientHeight/2),behavior:'instant'});}else viewport.scrollTo({left:0,top:0,behavior:'instant'});}
    function clearMotion(){generation++;animations.forEach(a=>a.cancel());animations=[];tokens.replaceChildren();moving=false;clearTimeout(timer);timer=null;}
    function token(value,points,delay,duration){const g=svgEl('g',{class:'flow-token'});g.style.opacity=0;g.append(svgEl('rect',{x:-31,y:-16,width:62,height:32,rx:16}),svgEl('text',{'text-anchor':'middle',y:6},String(value)));tokens.append(g);const frames=points.map(([x,y])=>({transform:`translate(${x}px, ${y}px)`,opacity:1}));const a=g.animate(frames,{duration,delay,fill:'forwards',easing:'ease-in-out'});animations.push(a);return a;}
    function transition(){clearMotion();render(false);focusBlock();const d=data(),i=step-3,duration=1700,version=generation;
      if(reduced.matches||step===0){render();if(running)hold();return;}
      moving=true;buttons.play.textContent='Pause';buttons.play.setAttribute('aria-pressed','true');
      const point=k=>[blocks[k].x,blocks[k].y];
      const outlet=k=>narrow()?[blocks[k].x,blocks[k].y+43]:[blocks[k].x+Number(blocks[k].g.firstChild.getAttribute('width'))/2,blocks[k].y];
      const inlet=k=>narrow()?[blocks[k].x,blocks[k].y-43]:[blocks[k].x-Number(blocks[k].g.firstChild.getAttribute('width'))/2,blocks[k].y];
      if(step===1||step===2){const input=step===1?x:w,converted=step===1?d.qx:d.qw,target=step===1?'qx':'qw',source=step===1?'x':'w';input.forEach((v,i)=>{const a=token(v,[point(source),point(target)],i*750,650);a.finished.then(()=>{if(!dead&&generation===version){a.effect.target.remove();blocks[target].value.textContent=converted.slice(0,i+1).join(', ');}}).catch(()=>{});});}
      else if(step<=5){
        const a=token(d.qx[i],narrow()?[point('qx'),[80,245],[165,245],point('mul')]:[point('qx'),[325,90],[325,190],point('mul')],0,600);
        const b=token(d.qw[i],narrow()?[point('qw'),[250,245],[165,245],point('mul')]:[point('qw'),[325,290],[325,190],point('mul')],600,600);
        a.finished.then(()=>{if(!dead&&generation===version){a.effect.target.remove();blocks.mul.value.textContent=`x: ${d.qx[i]-3}`;}}).catch(()=>{});
        b.finished.then(()=>{if(!dead&&generation===version){b.effect.target.remove();blocks.mul.value.textContent=String(d.terms[i]);}}).catch(()=>{});
        token(d.terms[i],[outlet('mul'),inlet('acc')],1250,650);
      }
      else if(step===6)token(d.sum,[outlet('acc'),inlet('req')],0,1100);
      else token(d.out,narrow()?[point('req'),[165,650],[77.5,650],point('act'),point('out')]:[point('req'),point('act'),point('out')],0,duration);
      Promise.all(animations.map(a=>a.finished)).then(()=>{if(dead||generation!==version)return;animations.forEach(a=>a.cancel());animations=[];tokens.replaceChildren();moving=false;render();if(running)hold();}).catch(()=>{});
    }
    function hold(){clearTimeout(timer);timer=setTimeout(()=>{if(!running||dead)return;if(step===7){running=false;render();return;}step++;transition();},2100);}
    function pause(){running=false;clearTimeout(timer);timer=null;animations.forEach(a=>a.pause());buttons.play.textContent=step===7&&!moving?'Replay':'Play';buttons.play.setAttribute('aria-pressed','false');}
    function seek(s,animate=false){clearMotion();running=false;step=s;render();if(animate)transition();else focusBlock();}
    on(buttons.play,'click',()=>{if(running||moving&&animations.some(a=>a.playState==='running'))return pause();running=true;if(moving){animations.forEach(a=>a.play());buttons.play.textContent='Pause';buttons.play.setAttribute('aria-pressed','true');return;}if(step===7)step=0;render();hold();});
    on(buttons.next,'click',()=>seek(Math.min(7,step+1),true));on(buttons.back,'click',()=>seek(Math.max(0,step-1)));on(buttons.restart,'click',()=>seek(0));on(menu,'change',()=>seek(Number(menu.value)));on(scaleMenu,'change',()=>{scale=Number(scaleMenu.value);seek(step);});
    on(root,'mlsys-view-change',e=>{if(e.detail?.view!=='explore')pause();});on(document,'visibilitychange',()=>{if(document.hidden)pause();});on(window,'beforeprint',pause);on(reduced,'change',()=>{if(reduced.matches&&moving){clearMotion();render();if(running)hold();}});
    on(window,'resize',()=>{pause();clearMotion();render();focusBlock();});
    const observer=new IntersectionObserver(entries=>{if(!entries[0].isIntersecting)pause();});observer.observe(root);
    render();return{pause,destroy(){dead=true;pause();clearMotion();observer.disconnect();abort.abort();root.replaceChildren();}};
  };
})();

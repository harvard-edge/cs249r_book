/* One output of a 1×3 by 3×1 GEMM: numerical toy, not a benchmark. */
(() => {
  'use strict';
  window.MLSysFigureLessons = window.MLSysFigureLessons || {};
  window.MLSysFigureLessons['quantized-matmul'] = root => {
    const abort = new AbortController();
    const on = (node, name, fn) => node.addEventListener(name, fn, {signal:abort.signal});
    const el = (tag, cls, text) => { const n=document.createElement(tag); if(cls)n.className=cls; if(text!==undefined)n.textContent=text; return n; };
    const x=[0.31,-0.52,0.77], w=[0.42,-0.18,0.24], sw=0.05, sy=0.02, zx=3, zw=0, zy=-4;
    const round = n => Math.sign(n)*Math.floor(Math.abs(n)+0.5);
    const clamp = n => Math.max(-128,Math.min(127,n));
    const fmt = n => String(Number(n.toFixed(6)));
    const stages=['Floating operands','Quantize activations','Quantize weights','Multiply pair 1','Multiply pair 2','Multiply pair 3','Requantize the sum','Activate and compare'];
    let stage=0, sx=0.1, running=false, timer=null, destroyed=false;
    root.classList.add('mlsys-qmm');
    const intro=el('p','mlsys-qmm-intro','Follow one output through the original pipeline. This declared toy multiplies a 1 × 3 activation row by a 3 × 1 weight column, with no bias.');
    const controls=el('div','mlsys-qmm-controls'), buttons={};
    for(const [key,label] of [['play','Play'],['back','Back'],['next','Next'],['restart','Restart']]) {const b=el('button','',label);b.type='button';buttons[key]=b;controls.append(b);}
    const stepLabel=el('label','','Step '), menu=el('select'); menu.id=root.id+'-sequence';stepLabel.htmlFor=menu.id;
    stages.forEach((s,i)=>{const o=el('option','',`${i+1}. ${s}`);o.value=i;menu.append(o);});stepLabel.append(menu);controls.append(stepLabel);
    const scenarioLabel=el('label','mlsys-qmm-scenario','Activation scale '), scenario=el('select');scenario.id=root.id+'-scale';scenarioLabel.htmlFor=scenario.id;
    for(const scale of [0.1,0.05]) {const o=el('option','',String(scale));o.value=scale;scenario.append(o);}scenarioLabel.append(scenario);
    const status=el('div','mlsys-qmm-status');status.setAttribute('aria-live','polite');status.setAttribute('aria-atomic','true');
    const heading=el('h4',''), narration=el('p','');status.append(heading,narration);
    const pipeline=el('div','mlsys-qmm-pipeline');
    const operands=el('div','mlsys-qmm-operands'), compute=el('section','mlsys-qmm-compute'), accumulator=el('section','mlsys-qmm-accumulator'), output=el('section','mlsys-qmm-output');
    const inputs=[];
    for(const name of ['Activation x','Weight w']) {
      const card=el('section','mlsys-qmm-operand'), title=el('h5','',name), meta=el('p','mlsys-qmm-meta'), row=el('div','mlsys-qmm-values'), cells=[];
      for(let i=0;i<3;i++){const c=el('div','mlsys-qmm-cell'),value=el('strong'),detail=el('small');c.append(value,detail);row.append(c);cells.push({c,value,detail});}
      card.append(title,meta,row);operands.append(card);inputs.push({card,meta,cells});
    }
    const compTitle=el('h5','','Integer matrix multiplication'), products=el('div','mlsys-qmm-products');compute.append(compTitle,products);
    const accTitle=el('h5','','INT32 accumulator'), sum=el('strong','mlsys-qmm-sum'), accDetail=el('p');accumulator.append(accTitle,sum,accDetail);
    const outTitle=el('h5','','Requantize → activation → INT8'), outValue=el('strong','mlsys-qmm-result'), outDetail=el('p');output.append(outTitle,outValue,outDetail);
    pipeline.append(operands,compute,accumulator,output);
    const equation=el('div','mlsys-qmm-equation'), comparison=el('p','mlsys-qmm-comparison'), storage=el('p','mlsys-qmm-storage');
    const assumptions=el('p','mlsys-qmm-assumptions','Signed INT8 range [−128, 127]. Round to nearest, ties away from zero; saturate after rounding. Chosen scales avoid clipping here. Nonzero zero-points must be subtracted before multiplication. Activation is ReLU for this worked example; the positive output passes unchanged.');
    root.append(intro,controls,scenarioLabel,status,pipeline,equation,comparison,storage,assumptions);
    function data(){
      const qx=x.map(v=>clamp(round(v/sx)+zx)), qw=w.map(v=>clamp(round(v/sw)+zw));
      const terms=qx.map((v,i)=>(v-zx)*(qw[i]-zw));
      const count=Math.max(0,Math.min(3,stage-2)), acc=terms.slice(0,count).reduce((a,b)=>a+b,0), total=terms.reduce((a,b)=>a+b,0);
      const qy=clamp(round(total*sx*sw/sy)+zy), activated=Math.max(zy,qy), reconstructed=(activated-zy)*sy, reference=x.reduce((a,v,i)=>a+v*w[i],0);
      return {qx,qw,terms,count,acc,total,qy,activated,reconstructed,reference};
    }
    function render(){
      const d=data();root.dataset.stage=stage;root.dataset.accumulator=d.acc;root.dataset.finalAccumulator=d.total;root.dataset.output=d.activated;root.dataset.reconstructed=fmt(d.reconstructed);
      heading.textContent=`${stage+1} / ${stages.length} · ${stages[stage]}`; menu.value=stage;scenario.value=sx;
      inputs.forEach((a,j)=>{
        const quantized=stage>=(j===0?1:2), source=j===0?x:w, qs=j===0?d.qx:d.qw, scale=j===0?sx:sw, zero=j===0?zx:zw;
        a.card.classList.toggle('mlsys-qmm-quantized',quantized);
        a.meta.textContent=quantized?`INT8 · scale s = ${scale}, zero-point z = ${zero}`:'FP32 · 3 values';
        a.cells.forEach((c,i)=>{c.value.textContent=quantized?qs[i]:fmt(source[i]);c.detail.textContent=quantized?`${fmt(source[i])} → ${qs[i]} → ${fmt((qs[i]-zero)*scale)}`:`element ${i}`;c.c.classList.toggle('mlsys-qmm-current',stage>=3&&stage<=5&&i===stage-3);});
      });
      products.replaceChildren();
      for(let i=0;i<3;i++){
        const line=el('div','mlsys-qmm-product');line.classList.toggle('mlsys-qmm-current',stage===i+3);
        line.textContent=i<d.count?`(${d.qx[i]} − ${zx}) × (${d.qw[i]} − ${zw}) = ${d.terms[i]}`:`Pair ${i+1}: waiting`;products.append(line);
      }
      compute.classList.toggle('mlsys-qmm-active',stage>=3&&stage<=5);accumulator.classList.toggle('mlsys-qmm-active',stage>=3&&stage<=5);output.classList.toggle('mlsys-qmm-active',stage>=6);
      sum.textContent=stage>=3?d.acc:'0';accDetail.textContent=`${d.count} / 3 products accumulated · 4-byte register`;
      outValue.textContent=stage>=6?`${stage===7?d.activated:d.qy} · INT8`:'Waiting for completed sum';
      outDetail.textContent=stage>=6?`Multiplier (sₓ × s𝓌) / sᵧ = ${fmt(sx*sw/sy)}. ${stage===7?'ReLU leaves this positive represented value unchanged.':'Apply scale, round, add output zero-point −4, then clamp.'}`:'The accumulator is not an INT8 output. Requantize only after all three contributions arrive.';
      const description=[
        'The two input paths begin in floating point. Keep the same three pairs throughout; the exact floating-point dot product is 0.4086.',
        `Quantize x using q = clamp(round(x / ${sx}) + 3). Each cell now shows original value → stored integer → reconstructed value; the representation has changed.`,
        'Quantize w with scale 0.05 and zero-point 0. Both operands are now integer arrays. Their scale and zero-point metadata are still required to interpret them.',
        `Read the first pair, subtract its zero-points, and multiply. The accumulator changes from 0 to ${d.terms[0]}; neither stored integer alone is the original real value.`,
        `Add the second product ${d.terms[1]} to the previous sum ${d.terms[0]}. A reduction retains its partial sum instead of emitting an output for each pair.`,
        `Add the third product ${d.terms[2]}. The complete sum is ${d.total}. INT32 accumulation protects a longer reduction from the limited INT8 range; ${d.total>127?'this sum already exceeds signed INT8’s maximum of 127.':'this tiny sum happens to fit INT8.'}`,
        `Rescale the complete INT32 sum into output units: round(${d.total} × ${fmt(sx*sw/sy)}) − 4 = ${d.qy}. This conversion changes scale and rounds; it is not an 8-bit truncation.`,
        `The output integer ${d.activated} reconstructs to ${fmt(d.reconstructed)}. Compared with the floating result 0.4086, signed error is ${fmt(d.reconstructed-d.reference)}. Try the other activation scale and replay: the arithmetic and rounding change together.`
      ];narration.textContent=description[stage];
      equation.textContent=stage<1?'Reference: 0.31 × 0.42 + (−0.52) × (−0.18) + 0.77 × 0.24 = 0.4086':stage<3?`Quantizer: q = clamp(round(r / s) + z). Reconstruction: r̂ = s × (q − z).` :stage<6?`INT32 partial sum: ${d.terms.slice(0,d.count).join(' + ')} = ${d.acc}`:`Output: (${d.activated} − (${zy})) × ${sy} = ${fmt(d.reconstructed)}`;
      comparison.hidden=stage!==7;comparison.textContent=`FP reference ${fmt(d.reference)} → reconstructed ${fmt(d.reconstructed)} · signed error ${fmt(d.reconstructed-d.reference)} · absolute error ${fmt(Math.abs(d.reconstructed-d.reference))}`;
      storage.textContent=`Raw operand values: ${stage>=2?'6 bytes as INT8':stage===1?'15 bytes: 3 INT8 activations + 3 FP32 weights':'24 bytes as FP32 before conversion'}; INT32 accumulator: 4 bytes. Scales and zero-points, output storage, and runtime workspace are separate. This is a storage accounting example, not a latency or total-memory speedup claim.`;
      buttons.back.disabled=stage===0;buttons.next.disabled=stage===7;buttons.play.textContent=running?'Pause':stage===7?'Replay':'Play';
    }
    function pause(){running=false;clearTimeout(timer);timer=null;if(!destroyed)render();}
    function schedule(){timer=setTimeout(()=>{if(!running||destroyed)return;stage++;if(stage===7)running=false;render();if(running)schedule();},5200);}
    on(buttons.play,'click',()=>{if(running)return pause();if(stage===7)stage=0;running=true;render();schedule();});
    on(buttons.back,'click',()=>{pause();stage=Math.max(0,stage-1);render();});on(buttons.next,'click',()=>{pause();stage=Math.min(7,stage+1);render();});
    on(buttons.restart,'click',()=>{pause();stage=0;render();});on(menu,'change',()=>{const v=Number(menu.value);pause();stage=v;render();});
    on(scenario,'change',()=>{const v=Number(scenario.value);pause();sx=v;render();});on(document,'visibilitychange',()=>{if(document.hidden)pause();});
    const observer=typeof IntersectionObserver==='undefined'?null:new IntersectionObserver(entries=>{if(!entries[0].isIntersecting&&running)pause();},{threshold:.1});observer?.observe(root);
    render();return {pause,destroy(){if(destroyed)return;pause();destroyed=true;observer?.disconnect();abort.abort();root.replaceChildren();}};
  };
})();

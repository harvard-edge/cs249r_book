/* Retains the exact canonical plot asset; the cursor follows worker count.
 * The readiness trace below is a qualitative single-batch dependency sequence,
 * not a sampled queue, time scale, benchmark, or alternative numeric registry.
 */
(()=>{'use strict';window.MLSysFigureLessons=window.MLSysFigureLessons||{};
  window.MLSysFigureLessons['dataloader-choke-point']=(root,context={})=>{
    const stages=['Inspect CPU supply','Prepare on host','Queue a ready batch','Transfer to device','Compute after readiness','Inspect capacity crossing'];

    const abort=new AbortController(),on=(e,t,f)=>e.addEventListener(t,f,{signal:abort.signal});
    const make=(tag,cls,text)=>{const e=document.createElement(tag);if(cls)e.className=cls;if(text!==undefined)e.textContent=text;return e};
    let step=0,running=false,timer=null,destroyed=false;
    const controls=make('div','mlsys-loader-controls'),buttons={};
    ['Play','Back','Next','Restart'].forEach(name=>{const b=make('button','',name);b.type='button';buttons[name.toLowerCase()]=b;controls.append(b)});
    const label=make('label','','Step '),menu=make('select');menu.id=root.id+'-sequence';label.htmlFor=menu.id;
    stages.forEach((title,i)=>{const option=make('option','',`${i+1}. ${title}`);option.value=i;menu.append(option)});label.append(menu);controls.append(label);
    const status=make('div','mlsys-loader-status');status.setAttribute('aria-live','polite');status.setAttribute('aria-atomic','true');
    function pause(){running=false;clearTimeout(timer);timer=null;if(!destroyed)render()}
    function tick(){timer=setTimeout(()=>{if(!running||destroyed)return;step++;if(step===stages.length-1)running=false;render();if(running)tick()},6500)}
    on(buttons.play,'click',()=>{if(running){pause();return}if(step===stages.length-1)step=0;running=true;render();tick()});
    on(buttons.back,'click',()=>{pause();step=Math.max(0,step-1);render()});on(buttons.next,'click',()=>{pause();step=Math.min(stages.length-1,step+1);render()});
    on(buttons.restart,'click',()=>{pause();step=0;render()});on(menu,'change',()=>{const requested=Number(menu.value);pause();step=requested;render()});
    on(document,'visibilitychange',()=>{if(document.hidden)pause()});
    const observer=typeof IntersectionObserver==='undefined'?null:new IntersectionObserver(e=>{if(!e[0].isIntersecting&&running)pause()},{threshold:.15});observer?.observe(root);
    function controlsState(){menu.value=step;buttons.back.disabled=step===0;buttons.next.disabled=step===stages.length-1;buttons.play.textContent=running?'Pause':step===stages.length-1?'Replay':'Play'}

    const intro=make('p','mlsys-loader-intro','Inspect the original worker-scaling plot, then follow one batch’s readiness dependencies. Storage and transfer are nonbinding in this chapter scenario. The batch trace illustrates ordering only: its states are not measured timings or steady-state queue lengths.');
    const plot=make('div','mlsys-loader-plot');
    const staticNodes=context.staticNodes||[];
    let source=null;
    for(const node of staticNodes){if(node.nodeType!==1)continue;source=node.matches?.('img')?node:node.querySelector?.('img');if(source)break}
    if(!source)source=context.figure?.querySelector('img')||context.outer?.querySelector('img');
    if(!source)throw new Error('Dataloader lesson requires its canonical plot image.');
    const image=source.cloneNode(true);image.removeAttribute('id');image.alt='Original DataLoader CPU-supply curve, red dashed GPU-capacity line, and throughput axes.';plot.append(image);
    const cursor=make('div','mlsys-loader-cursor');cursor.setAttribute('aria-hidden','true');plot.append(cursor);
    const chooser=make('label','mlsys-loader-workers','Inspect worker count '),workers=make('select');workers.id=root.id+'-workers';chooser.htmlFor=workers.id;
    for(let i=1;i<=16;i++){const o=make('option','',String(i));o.value=i;workers.append(o)}let count=4,previousStep=-1;workers.value=count;chooser.append(workers);
    const cue=make('p','mlsys-loader-cue');
    const trace=make('div','mlsys-loader-trace'),areas=[];
    for(const title of ['CPU preparation','Host ready queue','Host → device transfer','GPU execution']){const section=make('section','mlsys-loader-area');section.append(make('h4','',title));const body=make('div','mlsys-loader-state');section.append(body);areas.push(body);trace.append(section)}
    const footnote=make('p','mlsys-loader-footnote','One batch identity, B, moves only after preparation and transfer complete. Worker selection inspects the plotted capacity relationship; it does not change this dependency sequence. More workers cannot repair a binding storage or interconnect stage outside the stated scenario.');
    root.append(intro,controls,chooser,plot,cue,trace,status,footnote);
    on(workers,'change',()=>{const requested=Number(workers.value);pause();count=requested;render()});
    const narration=[
      'The blue line is CPU supply, not necessarily achieved training throughput. The limiting rate is the lower of CPU supply and the red dashed GPU capacity.',
      'Batch B is still being decoded and transformed on the host. It is not yet eligible for device transfer or GPU computation.',
      'Preparation completes. Batch B occupies the host ready queue; readiness does not mean its data is already resident on the accelerator.',
      'Transfer consumes the ready batch. Batch B is in flight; computation that needs it must wait until device readiness is established.',
      'Transfer has completed and batch B is resident. The GPU can now compute on it. Work on other batches may overlap these dependencies.',
      'Inspect the crossing near nine workers. Beyond it, additional CPU supply cannot raise this accelerator’s throughput. The CPU curve’s higher plateau is a supply cap, not a higher achieved training rate.'
    ];
    function render(){
      if(step===5&&previousStep!==5)count=9;
      previousStep=step;workers.value=count;
      // Plot geometry follows the reviewed source image: xlim 0.25…16.75,
      // axes extent x=12.8…98.2%; this is a worker cursor, not a fabricated rate.
      cursor.style.left=(12.8+(count-.25)/16.5*85.4)+'%';
      cue.textContent=`${count} worker${count===1?'':'s'}: ${count<9?'CPU supply below GPU capacity; input-starved regime.':'CPU supply can meet GPU capacity; accelerator-limited regime in this scenario.'}`;
      const batchStage=step===5?4:step;
      const text=batchStage===0?['Idle for this batch','Empty','Not started','Waiting for input']:batchStage===1?['B · preparing','Empty','Not started','Waiting for B']:batchStage===2?['B · preparation done','B · ready on host','Not started','Waiting for B']:batchStage===3?['Preparation done','B dequeued','B · in flight','Waiting for transfer']:['Preparation done','B dequeued','B · transfer complete','B · resident, computing'];
      areas.forEach((area,i)=>{area.textContent=text[i];area.classList.toggle('mlsys-loader-active',(batchStage===1&&i===0)||(batchStage===2&&i===1)||(batchStage===3&&i===2)||(batchStage===4&&i===3))});
      status.textContent=narration[step];controlsState();
    }

    render();
    return {pause,destroy(){if(destroyed)return;pause();destroyed=true;observer?.disconnect();abort.abort();root.replaceChildren()}};
  };
})();

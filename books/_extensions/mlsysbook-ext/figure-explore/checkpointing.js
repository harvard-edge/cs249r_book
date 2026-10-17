/* Source-faithful five-position, two-row checkpoint schematic.
 * Supplementary schedule: a0 is the input; each of four toy layers' backward
 * rules needs both its input and output activation. Counts exclude all other
 * training state. No time/byte/total-memory measurement is inferred.
 */
(()=>{'use strict';window.MLSysFigureLessons=window.MLSysFigureLessons||{};
  window.MLSysFigureLessons['checkpointing']=(root,context={})=>{
    const stages=['Keep input checkpoint','Forward to a1','Forward to a2','Keep a3 checkpoint','Finish forward and loss','Recompute a4','Backward layer 4','Recompute a1','Recompute a2','Backward layer 3','Backward layer 2','Backward layer 1'];

    const abort=new AbortController(),on=(e,t,f)=>e.addEventListener(t,f,{signal:abort.signal});
    const make=(tag,cls,text)=>{const e=document.createElement(tag);if(cls)e.className=cls;if(text!==undefined)e.textContent=text;return e};
    let step=0,running=false,timer=null,destroyed=false;
    const controls=make('div','mlsys-checkpoint-controls'),buttons={};
    ['Play','Back','Next','Restart'].forEach(name=>{const b=make('button','',name);b.type='button';buttons[name.toLowerCase()]=b;controls.append(b)});
    const label=make('label','','Step '),menu=make('select');menu.id=root.id+'-sequence';label.htmlFor=menu.id;
    stages.forEach((title,i)=>{const option=make('option','',`${i+1}. ${title}`);option.value=i;menu.append(option)});label.append(menu);controls.append(label);
    const status=make('div','mlsys-checkpoint-status');status.setAttribute('aria-live','polite');status.setAttribute('aria-atomic','true');
    function pause(){running=false;clearTimeout(timer);timer=null;if(!destroyed)render()}
    function tick(){timer=setTimeout(()=>{if(!running||destroyed)return;step++;if(step===stages.length-1)running=false;render();if(running)tick()},6500)}
    on(buttons.play,'click',()=>{if(running){pause();return}if(step===stages.length-1)step=0;running=true;render();tick()});
    on(buttons.back,'click',()=>{pause();step=Math.max(0,step-1);render()});on(buttons.next,'click',()=>{pause();step=Math.min(stages.length-1,step+1);render()});
    on(buttons.restart,'click',()=>{pause();step=0;render()});on(menu,'change',()=>{const requested=Number(menu.value);pause();step=requested;render()});
    on(document,'visibilitychange',()=>{if(document.hidden)pause()});
    const observer=typeof IntersectionObserver==='undefined'?null:new IntersectionObserver(e=>{if(!e[0].isIntersecting&&running)pause()},{threshold:.15});observer?.observe(root);
    function controlsState(){menu.value=step;buttons.back.disabled=step===0;buttons.next.disabled=step===stages.length-1;buttons.play.textContent=running?'Pause':step===stages.length-1?'Replay':'Play'}

    const intro=make('p','mlsys-checkpoint-intro','The same five positions and two pass directions as Diagram. Supplementary toy schedule: a0 is the input, layers 1–4 produce a1–a4, and each backward rule needs that layer’s input and output. Labels expose lifetimes; activation sizes are not specified.');
    const diagram=make('div','mlsys-checkpoint-diagram'),rows=[];
    for(const [r,title] of ['Forward Pass →','← Backward Pass'].entries()){
      const section=make('section','mlsys-checkpoint-row');section.append(make('h4','',title));
      const chain=make('div','mlsys-checkpoint-chain'),nodes=[];
      for(let i=0;i<5;i++){
        const wrapper=make('div','mlsys-checkpoint-position');
        const node=make('div','mlsys-checkpoint-node',`a${i}`),state=make('span','mlsys-checkpoint-state');
        wrapper.append(node,state);chain.append(wrapper);nodes.push({node,state});
        if(i<4)chain.append(make('span','mlsys-checkpoint-arrow',r===0?'→':'←'));
      }
      section.append(chain);diagram.append(section);rows.push(nodes);
    }
    const tally=make('p','mlsys-checkpoint-tally');
    const legend=make('p','mlsys-checkpoint-footnote','Green: retained checkpoint · dashed gray: not retained · orange: recomputed and live · outlined: current operation. Top row records the forward schedule; bottom row shows current backward residency.');
    const caveat=make('p','mlsys-checkpoint-footnote','Counts cover only the five illustrated activation tensors, including the input. Weights, gradients, optimizer state, loss state, and workspaces are excluded. This small schedule is not a claim about total-memory savings or wall-clock speedup.');
    root.append(intro,controls,diagram,legend,tally,status,caveat);
    const states=[
      {live:[0],done:0,text:'Keep a0 as a checkpoint. No layer output exists yet.'},
      {live:[0,1],done:1,text:'Layer 1 creates a1. It must remain live until layer 2 consumes it.'},
      {live:[0,2],done:2,text:'Layer 2 consumes a1 and creates a2. Release a1; it is not a retained checkpoint.'},
      {live:[0,3],done:3,text:'Layer 3 consumes a2 and creates checkpoint a3. Release a2 after that forward consumer finishes.'},
      {live:[0,3],done:4,text:'Layer 4 creates a4 and the loss consumes it. Discard a4 afterward. Only a0 and a3 remain at the forward boundary.'},
      {live:[0,3,4],done:4,recomputed:[4],current:4,text:'Replay layer 4 from a3 to reconstruct a4. Keep both until backward layer 4 consumes them.'},
      {live:[0,3],done:4,current:4,text:'Backward layer 4 consumes a3 and a4. Release a4; a3 still has a backward-layer-3 consumer.'},
      {live:[0,1,3],done:4,recomputed:[1],current:1,text:'Replay layer 1 from a0. Retain reconstructed a1 for layer 2’s forward replay and its later backward consumers.'},
      {live:[0,1,2,3],done:4,recomputed:[1,2],current:2,text:'Replay layer 2 to reconstruct a2. Keep a1 and a2; do not release a1 after replay because backward layer 2 still needs it.'},
      {live:[0,1,2],done:4,recomputed:[1,2],current:3,text:'Backward layer 3 consumes a2 and checkpoint a3. Release a3 after its last consumer. Keep a2 for backward layer 2.'},
      {live:[0,1],done:4,recomputed:[1],current:2,text:'Backward layer 2 consumes a1 and a2. Release a2, but retain a1 for backward layer 1.'},
      {live:[],done:4,current:1,text:'Backward layer 1 consumes a0 and a1. Release both: this toy chain has no remaining activation consumers. Three extra layer forwards reconstructed a4, a1, and a2.'}
    ];
    function render(){const s=states[step];
      rows[0].forEach(({node,state},i)=>{
        node.className='mlsys-checkpoint-node '+(i>s.done?'mlsys-checkpoint-pending':i===0||i===3?'mlsys-checkpoint-kept':'mlsys-checkpoint-dropped')+(step<4&&i===s.done?' mlsys-checkpoint-current':'');
        state.textContent=i>s.done?'future':i===0||i===3?'saved':step<4&&s.live.includes(i)?'live, transient':'dropped';
      });
      rows[1].forEach(({node,state},i)=>{
        const active=step>=5&&s.live.includes(i),recomputed=active&&s.recomputed?.includes(i);
        node.className='mlsys-checkpoint-node '+(active?(recomputed?'mlsys-checkpoint-recomputed':'mlsys-checkpoint-kept'):'mlsys-checkpoint-dropped')+(step>=5&&s.current===i?' mlsys-checkpoint-current':'');
        state.textContent=step<5?'future':active?(recomputed?'rebuilt, live':'saved, live'):'not live';
      });
      tally.textContent=`Current live illustrated activations: ${s.live.length}/5 (${s.live.length?s.live.map(i=>'a'+i).join(', '):'none'}).`;
      status.textContent=s.text;controlsState();
    }

    render();
    return {pause,destroy(){if(destroyed)return;pause();destroyed=true;observer?.disconnect();abort.abort();root.replaceChildren()}};
  };
})();

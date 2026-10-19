(() => {
'use strict';window.MLSysFigureLessons=window.MLSysFigureLessons||{};
window.MLSysFigureLessons["v3-intent-ordering"]=(root,context)=>{
const abort=new AbortController(),on=(el,event,fn)=>el.addEventListener(event,fn,{signal:abort.signal});
const el=(tag,cls,text)=>{const n=document.createElement(tag);if(cls)n.className=cls;if(text!==undefined)n.textContent=text;return n;};
root.classList.add('mlsys-v3');let step=0,running=false,timer=null,dead=false;
const controls=el('div','v3-controls'),buttons={};for(const name of ['Play','Back','Next','Restart']){const b=el('button','',name);b.type='button';buttons[name.toLowerCase()]=b;controls.append(b);}
const sequence=el('select'),lab=el('label','','Sequence ');sequence.id=(root.id||('v3-'+Math.random().toString(36).slice(2)))+'-sequence';lab.htmlFor=sequence.id;lab.append(sequence);controls.append(lab);root.append(controls);
const diagram=el('div','v3-diagram'),status=el('p','v3-status');status.setAttribute('aria-live','polite');status.setAttribute('aria-atomic','true');root.append(diagram,status);
const card=(title,text,cls='')=>{const n=el('section','v3-card '+cls);n.append(el('strong','',title),el('p','',text));return n;};
const grid=(items,cls='')=>{const g=el('div','v3-grid '+cls);items.forEach(n=>g.append(n));return g;};
let stages=[],draw=()=>{};

stages=['Validate proposal','Admit and reserve','Wait for intent acknowledgment','Cross durability barrier','Observation still pending','Record correlated outcome'];let branch='normal';
const selector=el('select'),caseLabel=el('label','','Recovery case ');selector.id=sequence.id+'-case';caseLabel.htmlFor=selector.id;for(const [v,t] of [['normal','Normal progress'],['lostack','Intent acknowledgment lost'],['crash','Crash after acknowledgment, before dispatch'],['timeout','Dispatch followed by timeout']]){const o=el('option','',t);o.value=v;selector.append(o);}caseLabel.append(selector);controls.append(caseLabel);on(selector,'change',()=>{pause();branch=selector.value;step=0;render();});
draw=()=>{diagram.replaceChildren();const stopped=branch==='lostack'&&step>=2||branch==='crash'&&step>=3;const ack=step>=2&&branch!=='lostack',sent=step>=3&&!stopped,settled=step===5&&branch==='normal';const current=stopped?(branch==='lostack'?2:3):step;
const events=el('div','v3-events');[['e_prop','proposal',0],['e_auth','admission',1],['e_flush','intent ack',2],['a_ext','dispatch',3],['e_obs','evidence',5]].forEach(([title,text,i])=>events.append(card(title,text+(i===3?(sent?' · sent':' · withheld'):i===2?(ack?' · received':' · not received'):i===5?(settled?' · correlated outcome':step>=4&&!stopped?' · pending / unknown':' · absent'):''),(i===current?'v3-active ':'')+(i===3?'v3-effect':''))));diagram.append(events,card('Durability barrier','Runtime withholds dispatch until the required intent acknowledgment is received under the declared failure domain.','v3-active'));
const state=!ack?'Not dispatched':settled?'Established':'In doubt';diagram.append(grid([card('Before acknowledgment','Not dispatched; intent may already be durable if acknowledgment was lost.',!ack?'v3-active':''),card('After acknowledgment, before adequate evidence','In doubt; withhold dependent work.',ack&&!settled?'v3-unknown':''),card('Adequate correlated evidence','Established, within endpoint contract.',settled?'v3-active v3-state':'')]),card('Recovery conclusion now',state+' · '+(!ack?'Inspect the complete authoritative prefix. Absence proves non-dispatch only with a complete prefix.':settled?'Recorded evidence establishes this outcome.':'Do not infer completion or authorize dependent work from missing evidence.')));
status.textContent=branch==='lostack'&&step>=2?'The intent may be durable, but its acknowledgment did not reach the worker. Dispatch remains withheld. Recovery must inspect the authoritative prefix.':branch==='crash'&&step>=3?'The worker crashes after receiving intent acknowledgment but before dispatch. Surviving intent alone leaves recovery in doubt; a logged admission does not prove an external effect.':branch==='timeout'&&step>=4?'Dispatch may have applied the remote effect. A timeout or pending observation is durable evidence of uncertainty, not an established outcome. Reconcile before dependent work.':[
'Validate the complete action proposal; no external request has been sent.',
'Admit authorization and reserve resources. Reservation is not dispatch or proof of effect.',
'Wait for the intent acknowledgment under the declared failure domain. Only then may dispatch cross the barrier.',
'Dispatch carries the request outward. Its outcome remains in doubt until adequate correlated evidence is recorded.',
'A pending observation can be recorded without establishing success or failure. Dependent work stays withheld.',
'Correlated outcome evidence is recorded. This normal case is now established within the endpoint contract, not merely because time passed.'
][step];};

stages.forEach((name,i)=>{const opt=el('option','',`${i+1}. ${name}`);opt.value=i;sequence.append(opt);});
function render(){draw();sequence.value=step;buttons.back.disabled=step===0;buttons.next.disabled=step===stages.length-1;buttons.play.textContent=running?'Pause':step===stages.length-1?'Replay':'Play';root.dataset.step=step;root.dataset.playing=String(running);}
function pause(){running=false;clearTimeout(timer);timer=null;if(!dead)render();}
function schedule(){timer=setTimeout(()=>{if(!running||dead)return;if(document.hidden||root.closest('[hidden]')){pause();return;}step++;if(step===stages.length-1)running=false;render();if(running)schedule();},6500);}
on(buttons.play,'click',()=>{if(running){pause();return;}if(step===stages.length-1)step=0;running=true;render();schedule();});
on(buttons.back,'click',()=>{pause();step=Math.max(0,step-1);render();});on(buttons.next,'click',()=>{pause();step=Math.min(stages.length-1,step+1);render();});on(buttons.restart,'click',()=>{pause();step=0;render();});on(sequence,'change',()=>{const n=Number(sequence.value);pause();step=n;render();});
on(document,'visibilitychange',()=>{if(document.hidden)pause();});
const obs=typeof IntersectionObserver==='undefined'?null:new IntersectionObserver(entries=>{if(!entries[0].isIntersecting&&running)pause();},{threshold:.08});obs?.observe(root);
const hiddenObs=new MutationObserver(()=>{if(running&&root.closest('[hidden]'))pause();});for(let n=root;n;n=n.parentElement)hiddenObs.observe(n,{attributes:true,attributeFilter:['hidden']});
render();return{pause,destroy(){pause();dead=true;abort.abort();obs?.disconnect();hiddenObs.disconnect();root.replaceChildren();}};
};})();

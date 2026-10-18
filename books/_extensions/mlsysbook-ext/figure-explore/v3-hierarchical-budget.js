(() => {
'use strict';window.MLSysFigureLessons=window.MLSysFigureLessons||{};
window.MLSysFigureLessons["v3-hierarchical-budget"]=(root,context)=>{
const abort=new AbortController(),on=(el,event,fn)=>el.addEventListener(event,fn,{signal:abort.signal});
const el=(tag,cls,text)=>{const n=document.createElement(tag);if(cls)n.className=cls;if(text!==undefined)n.textContent=text;return n;};
root.classList.add('mlsys-v3');let step=0,running=false,timer=null,dead=false;
const controls=el('div','v3-controls'),buttons={};for(const name of ['Play','Back','Next','Restart']){const b=el('button','',name);b.type='button';buttons[name.toLowerCase()]=b;controls.append(b);}
const sequence=el('select'),lab=el('label','','Sequence ');sequence.id=(root.id||('v3-'+Math.random().toString(36).slice(2)))+'-sequence';lab.htmlFor=sequence.id;lab.append(sequence);controls.append(lab);root.append(controls);
const diagram=el('div','v3-diagram'),status=el('p','v3-status');status.setAttribute('aria-live','polite');status.setAttribute('aria-atomic','true');root.append(diagram,status);
const card=(title,text,cls='')=>{const n=el('section','v3-card '+cls);n.append(el('strong','',title),el('p','',text));return n;};
const grid=(items,cls='')=>{const g=el('div','v3-grid '+cls);items.forEach(n=>g.append(n));return g;};
let stages=[],draw=()=>{};

stages=['Planning settled','Both allocations granted','Trace child closed','Fuzzing stopped; bill pending','Late bill settled; child closed'];
const balances=[[42,0,4958],[42,3700,1258],[422,2500,2078],[422,2500,2078],[1972,0,3028]];const money=n=>'$'+(n/1000).toFixed(3).replace(/0+$/,'').replace(/\.$/,'');
const account=(title,amounts,total,active=false)=>{const box=card(title,`Settled E ${money(amounts[0])} + reserved R ${money(amounts[1])} + free F ${money(amounts[2])} = ${money(total)}`,active?'v3-active':'');const bar=el('div','v3-bar');bar.setAttribute('role','img');bar.setAttribute('aria-label',box.textContent);amounts.forEach((v,i)=>{if(v){const s=el('span','v3-segment '+['v3-settled','v3-reserved','v3-free'][i]);s.style.width=(100*v/total)+'%';s.title=['Settled','Reserved','Free'][i]+' '+money(v);bar.append(s);}});box.append(bar);return box;};
draw=()=>{diagram.replaceChildren();diagram.append(el('p','v3-note','Root budget Cmax = E + R + F = $5. Values reproduce the chapter’s late-bill example; these are currency balances, not measured execution timing. Gray = settled; amber = reserved; green = free.'));
for(let i=0;i<=step;i++){const row=account(stages[i],balances[i],5000,i===step);if(i===1)row.append(el('p','v3-note','Reserved $3.70 = trace $1.20 + fuzzing $2.50. Allocations are drawn from the same root budget.'));if(i===2)row.append(el('p','v3-note','Trace settles $0.38 and returns its remaining $0.82. Root settled becomes $0.422.'));diagram.append(row);if(i===3){const child=account('Fuzzing child account at local stop',[850,1000,650],2500,true);child.append(el('p','v3-note','Root still reserves the entire $2.50 child allocation. Its $0.85 settled + $1 pending remote + $0.65 locally free must not also be spent by the root.'));diagram.append(child);}}
if(step===4)diagram.append(card('Settlement at child close','Late bill settles $0.70 of $1 pending. Child total settles $1.55; $0.95 returns. Root settled $0.422 + $1.55 = $1.972; free $2.078 + $0.95 = $3.028.','v3-state'));
const [e,r,f]=balances[step];diagram.append(card('Conservation check',`${money(e)} + ${money(r)} + ${money(f)} = $5. Root spendable now: ${money(f)}.`));
status.textContent=[
'Planning has settled $0.042. The remaining $4.958 is free at the root.',
'The root grants $1.20 to trace and $2.50 to fuzzing. Both consume one shared reservation pool; they do not each inherit a fresh $5 budget.',
'Trace closes with $0.38 charged. Its unused balance returns, while the entire fuzzing allocation remains reserved.',
'Local stop releases nothing to the root. A remote bill is still pending; the child’s internal balances explain why cancellation cannot free the full allocation.',
'After the late bill settles and the child closes, its total charge becomes settled at the root and only the unused $0.95 returns. The root budget remains exactly $5.'
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

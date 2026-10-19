(() => {
'use strict';window.MLSysFigureLessons=window.MLSysFigureLessons||{};
window.MLSysFigureLessons["v3-paged-mapping"]=(root,context)=>{
const abort=new AbortController(),on=(el,event,fn)=>el.addEventListener(event,fn,{signal:abort.signal});
const el=(tag,cls,text)=>{const n=document.createElement(tag);if(cls)n.className=cls;if(text!==undefined)n.textContent=text;return n;};
root.classList.add('mlsys-v3');let step=0,running=false,timer=null,dead=false;
const controls=el('div','v3-controls'),buttons={};for(const name of ['Play','Back','Next','Restart']){const b=el('button','',name);b.type='button';buttons[name.toLowerCase()]=b;controls.append(b);}
const sequence=el('select'),lab=el('label','','Sequence ');sequence.id=(root.id||('v3-'+Math.random().toString(36).slice(2)))+'-sequence';lab.htmlFor=sequence.id;lab.append(sequence);controls.append(lab);root.append(controls);
const diagram=el('div','v3-diagram'),status=el('p','v3-status');status.setAttribute('aria-live','polite');status.setAttribute('aria-atomic','true');root.append(diagram,status);
const card=(title,text,cls='')=>{const n=el('section','v3-card '+cls);n.append(el('strong','',title),el('p','',text));return n;};
const grid=(items,cls='')=>{const g=el('div','v3-grid '+cls);items.forEach(n=>g.append(n));return g;};
let stages=[],draw=()=>{};

stages=['Ten-token prefix','Resolve position 9','Two candidates share prefix','Allocate private tail 14','Copy initialized tail entries','Append B position 10'];
const slotBlock=(id,values,active=false,other=false,owned=true)=>{const state=other?'other':owned?'allocated':'free';const n=el('div','v3-block v3-block-'+state+(active?' v3-active':''));n.dataset.block=id;n.dataset.allocation=state;n.append(el('strong','',`Block ${id}`));const slots=el('div','v3-slots');for(let i=0;i<4;i++){const initialized=values[i]!==undefined;const slot=el('span','v3-slot'+(initialized?' v3-initialized':owned&&!other?' v3-allocated-unwritten':''),initialized?values[i]:other?'?':owned?'·':'—');slot.setAttribute('aria-label',`slot ${i}: ${initialized?'initialized position '+values[i]:other?'owned by another request':owned?'allocated, not written':'free, not allocated'}`);slots.append(slot);}n.append(slots,el('small','v3-block-owner',other?'Other request':owned?'Allocated':'Free pool'));return n;};
draw=()=>{diagram.replaceChildren();diagram.append(el('h4','','(a) Ten-token prefix · four-token blocks'));
const logical=grid([slotBlock('logical 0',[0,1,2,3]),slotBlock('logical 1',[4,5,6,7]),slotBlock('logical 2',[8,9],step===1)]);diagram.append(logical,card('Block table','logical 0 → physical 3  |  logical 1 → physical 7  |  logical 2 → physical 9','v3-state'));
const pool=el('div','v3-pool');for(let i=0;i<12;i++)pool.append(slotBlock(i,i===3?[0,1,2,3]:i===7?[4,5,6,7]:i===9?[8,9]:[],step===1&&i===9,[1,4,5,10].includes(i),[3,7,9].includes(i)));diagram.append(pool,el('p','v3-note','Green fill: initialized KV. Green outlined/shaded dots: allocated, not written. Dashed gray blocks with dashes: free pool, not allocated. Gray blocks with question marks: other requests. IDs are pool indices, not addresses.'));
diagram.append(card('Position 9 lookup','⌊9 / 4⌋ = 2 → table[2] = 9 → slot 9 mod 4 = 1',step===1?'v3-active':''),el('h4','','(b) Candidate B appends to a shared partial tail'));
const btable=step>=4?'[3, 7, 14]':'[3, 7, 9]';diagram.append(card('Candidate A block table','[3, 7, 9]'),grid([slotBlock(3,[0,1,2,3]),slotBlock(7,[4,5,6,7]),slotBlock(9,[8,9])]),card('Candidate B block table',step<2?'Not forked yet':btable,step>=3?'v3-active':''));
if(step>=2)diagram.append(card('Shared full blocks','A and B both reference blocks 3 and 7. No copy of these full blocks.','v3-state'));
if(step===3)diagram.append(card('Pending private tail','Block 14 is allocated to B, but B’s active table still points to block 9. Copy initialized entries before committing the new table entry.','v3-state'));
if(step>=3)diagram.append(slotBlock(14,step>=5?[8,9,10]:step>=4?[8,9]:[],true),el('p','v3-note',step<4?'Block 14 allocated; private tail is not ready for use until initialized entries are copied.':step===4?'Copied slots 0–1 from block 9 → block 14, then committed B table[2] = 14. Slots 2–3 remain allocated but unwritten.':'Write B position 10 into block 14, slot 2. A block 9 is unchanged.'));
status.textContent=[
'The ten-token prefix occupies physical blocks 3, 7, and 9. Logical order does not require physically adjacent storage.',
'Position 9 resolves through logical block 2 to physical block 9, slot 1. This is an entry offset, not a byte address.',
'A and B initially share the prefix, including its two-entry partial tail. B cannot privately append into A’s shared block.',
'Allocate block 14 for B’s private tail. It is reserved, not free, even though none of its entries are initialized. Keep B’s active table [3, 7, 9] until copying completes.',
'Copy initialized positions 8 and 9 into block 14 first, then commit B’s table [3, 7, 14]. Its two remaining slots are allocated but unwritten. Full blocks 3 and 7 remain shared.',
'Append position 10 only in B’s private block 14. A still sees [3, 7, 9]; B sees [3, 7, 14]. Sharing saves prefix copies, while the private append requires tail storage and copying.'
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

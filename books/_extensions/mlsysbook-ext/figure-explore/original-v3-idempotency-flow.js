/* Original chapter artwork with inspectable, endpoint-driven systems state. */
(() => {
'use strict';
const spec = {
  "viewBox": "0 0 760 370",
  "asset": "idempotency_nonce_execution_flow.svg",
  "question": "Why does a lost reply need a new attempt, but the same operation ID?",
  "scope": "Illustrative IDs op_A, a1/a2/a3, payload P, and retained result R on the original runtime/endpoint diagram. The endpoint authorizes each attempt within caller C/resource R and an active retention window. This example supports one atomic effect-and-result transaction.",
  "purpose": "Follow one operation through a committed effect, lost response, safe retry, and rejected payload conflict while preserving identity.",
  "guard": "Deduplication requires the endpoint contract and retention scope. It does not make arbitrary external effects transactional. A timeout leaves the client uncertain; a new attempt is not a new intent. Pending/unresolved branches remain reference alternatives, not extra executed effects.",
  "slots": {
    "intent": {
      "x": 135,
      "y": 91,
      "width": 175,
      "font": 11
    },
    "attempt": {
      "x": 135,
      "y": 148,
      "width": 175,
      "font": 13
    },
    "request": {
      "x": 135,
      "y": 166,
      "width": 175,
      "font": 11
    },
    "timeout": {
      "x": 135,
      "y": 245,
      "width": 175,
      "font": 11
    },
    "unresolved": {
      "x": 135,
      "y": 325,
      "width": 175,
      "font": 11
    },
    "retained": {
      "x": 535,
      "y": 82,
      "width": 115,
      "font": 11
    },
    "effect": {
      "x": 675,
      "y": 285,
      "width": 80,
      "font": 11
    },
    "result": {
      "x": 675,
      "y": 323,
      "width": 80,
      "font": 10
    },
    "conflict": {
      "x": 535,
      "y": 222,
      "width": 115,
      "font": 11
    }
  },
  "defaults": {},
  "fields": [
    "Logical operation",
    "Current attempt / payload",
    "Effect",
    "Completion record",
    "Client knowledge"
  ],
  "states": [
    {
      "label": "Admit one operation identity",
      "description": "Admit one illustrative operation op_A, bound to caller C, resource R, and payload P. Attempt IDs identify transmissions, while the operation ID identifies the intended effect.",
      "text": {
        "intent": "op_A · C / R · payload P",
        "attempt": "No attempt sent",
        "request": "op_A · payload P",
        "timeout": "not observed",
        "unresolved": "attempt budget not exhausted",
        "retained": "no retained completion",
        "effect": "not committed",
        "result": "not recorded",
        "conflict": "not selected"
      },
      "values": [
        "op_A",
        "a1 · P",
        "Not committed",
        "No completion record",
        "awaiting reply"
      ],
      "facts": {
        "operation": "op_A",
        "attempt": "a1",
        "payload": "P",
        "effect_committed": false,
        "result_recorded": false,
        "duplicate_effect": false,
        "response": "awaiting reply",
        "authorized": true
      },
      "camera": [
        135,
        180
      ]
    },
    {
      "label": "Send attempt a1",
      "description": "Send op_A with fresh attempt a1. When the request reaches lookup, the endpoint checks authorization and its scoped retention record before deciding whether to execute.",
      "text": {
        "intent": "op_A · C / R · payload P",
        "attempt": "Attempt a1",
        "request": "op_A · payload P",
        "timeout": "not observed",
        "unresolved": "attempt budget not exhausted",
        "retained": "no retained completion",
        "effect": "not committed",
        "result": "not recorded",
        "conflict": "not selected"
      },
      "values": [
        "op_A",
        "a1 · P",
        "Not committed",
        "No completion record",
        "awaiting reply"
      ],
      "facts": {
        "operation": "op_A",
        "attempt": "a1",
        "payload": "P",
        "effect_committed": false,
        "result_recorded": false,
        "duplicate_effect": false,
        "response": "awaiting reply",
        "authorized": true
      },
      "camera": [
        135,
        180
      ],
      "motion": {
        "label": "op_A / a1",
        "route": [
          [
            238,
            144
          ],
          [
            308,
            144
          ]
        ],
        "role": "tool",
        "lock_camera": false
      }
    },
    {
      "label": "Commit effect and result atomically",
      "description": "The operation is unseen. When this one transaction commits, the effect and retained result R become durable together. Neither is presented as committed before the transaction finishes.",
      "text": {
        "intent": "op_A · C / R · payload P",
        "attempt": "Attempt a1",
        "request": "op_A · payload P",
        "timeout": "not observed",
        "unresolved": "attempt budget not exhausted",
        "retained": "retained result R",
        "effect": "a1 effect committed",
        "result": "R retained",
        "conflict": "not selected"
      },
      "values": [
        "op_A",
        "a1 · P",
        "Committed once",
        "R durably retained",
        "awaiting reply"
      ],
      "facts": {
        "operation": "op_A",
        "attempt": "a1",
        "payload": "P",
        "effect_committed": true,
        "result_recorded": true,
        "duplicate_effect": false,
        "response": "awaiting reply",
        "authorized": true
      },
      "camera": [
        675,
        295
      ],
      "motion": {
        "label": "op_A",
        "route": [
          [
            370,
            154
          ],
          [
            450,
            154
          ],
          [
            450,
            290
          ],
          [
            600,
            290
          ],
          [
            618,
            290
          ],
          [
            675,
            285
          ],
          [
            675,
            323
          ]
        ],
        "role": "tool",
        "lock_camera": false
      }
    },
    {
      "label": "Lose the committed reply",
      "description": "The endpoint has committed, but the reply is lost. When the client times out, its outcome is unknown; the endpoint’s retained completion record still exists.",
      "text": {
        "intent": "op_A · C / R · payload P",
        "attempt": "Attempt a1",
        "request": "op_A · payload P",
        "timeout": "a1: effect unknown to client",
        "unresolved": "attempt budget not exhausted",
        "retained": "retained result R",
        "effect": "a1 effect committed",
        "result": "R retained",
        "conflict": "not selected"
      },
      "values": [
        "op_A",
        "a1 · P",
        "Committed once",
        "R durably retained",
        "unknown"
      ],
      "facts": {
        "operation": "op_A",
        "attempt": "a1",
        "payload": "P",
        "effect_committed": true,
        "result_recorded": true,
        "duplicate_effect": false,
        "response": "unknown",
        "authorized": true
      },
      "camera": [
        135,
        180
      ],
      "motion": {
        "label": "reply lost",
        "route": [
          [
            310,
            160
          ],
          [
            275,
            160
          ]
        ],
        "role": "fault",
        "lock_camera": false
      }
    },
    {
      "label": "Retry with fresh attempt a2",
      "description": "Send fresh attempt a2 with the same op_A, scope, and payload P. A transport retry must not create a new logical operation.",
      "text": {
        "intent": "op_A · C / R · payload P",
        "attempt": "Attempt a2",
        "request": "op_A · payload P",
        "timeout": "a1: effect unknown to client",
        "unresolved": "attempt budget not exhausted",
        "retained": "retained result R",
        "effect": "a1 effect committed",
        "result": "R retained",
        "conflict": "not selected"
      },
      "values": [
        "op_A",
        "a2 · P",
        "Committed once",
        "R durably retained",
        "unknown"
      ],
      "facts": {
        "operation": "op_A",
        "attempt": "a2",
        "payload": "P",
        "effect_committed": true,
        "result_recorded": true,
        "duplicate_effect": false,
        "response": "unknown",
        "authorized": true
      },
      "camera": [
        135,
        180
      ],
      "motion": {
        "label": "op_A / a2",
        "route": [
          [
            135,
            232
          ],
          [
            30,
            232
          ],
          [
            30,
            152
          ],
          [
            238,
            144
          ],
          [
            308,
            144
          ]
        ],
        "role": "tool",
        "lock_camera": false
      }
    },
    {
      "label": "Return the retained result",
      "description": "The completed lookup branch returns R. When the response arrives, the client can establish the recorded outcome without applying the effect again.",
      "text": {
        "intent": "op_A · C / R · payload P",
        "attempt": "Attempt a2",
        "request": "op_A · payload P",
        "timeout": "a1: effect unknown to client",
        "unresolved": "attempt budget not exhausted",
        "retained": "retained result R",
        "effect": "a1 effect committed",
        "result": "R retained",
        "conflict": "not selected"
      },
      "values": [
        "op_A",
        "a2 · P",
        "Committed once",
        "R durably retained",
        "retained R"
      ],
      "facts": {
        "operation": "op_A",
        "attempt": "a2",
        "payload": "P",
        "effect_committed": true,
        "result_recorded": true,
        "duplicate_effect": false,
        "response": "retained R",
        "authorized": true
      },
      "camera": [
        535,
        140
      ],
      "motion": {
        "label": "R",
        "route": [
          [
            535,
            70
          ],
          [
            450,
            70
          ],
          [
            450,
            160
          ],
          [
            310,
            160
          ],
          [
            240,
            160
          ],
          [
            135,
            160
          ]
        ],
        "role": "state",
        "lock_camera": false
      }
    },
    {
      "label": "Probe a mismatched payload",
      "description": "Send a separate malformed probe a3 with op_A but payload P2. This is not a valid mutation of the earlier intent; the endpoint must compare the bound payload.",
      "text": {
        "intent": "op_A · C / R · payload P",
        "attempt": "Attempt a3",
        "request": "op_A · payload P2",
        "timeout": "a1: effect unknown to client",
        "unresolved": "attempt budget not exhausted",
        "retained": "retained result R",
        "effect": "a1 effect committed",
        "result": "R retained",
        "conflict": "not selected"
      },
      "values": [
        "op_A",
        "a3 · P2",
        "Committed once",
        "R durably retained",
        "retained R"
      ],
      "facts": {
        "operation": "op_A",
        "attempt": "a3",
        "payload": "P2",
        "effect_committed": true,
        "result_recorded": true,
        "duplicate_effect": false,
        "response": "retained R",
        "authorized": true
      },
      "camera": [
        135,
        180
      ],
      "motion": {
        "label": "P2 / a3",
        "route": [
          [
            238,
            144
          ],
          [
            308,
            144
          ]
        ],
        "role": "tool",
        "lock_camera": false
      }
    },
    {
      "label": "Reject the ID conflict",
      "description": "When the payload comparison finishes, return an ID conflict. The original effect and retained result R remain unchanged; no second effect is executed.",
      "text": {
        "intent": "op_A · C / R · payload P",
        "attempt": "Attempt a3",
        "request": "op_A · payload P2",
        "timeout": "a1: effect unknown to client",
        "unresolved": "attempt budget not exhausted",
        "retained": "retained result R",
        "effect": "a1 effect committed",
        "result": "R retained",
        "conflict": "P2 differs: reject"
      },
      "values": [
        "op_A",
        "a3 · P2",
        "Committed once",
        "R durably retained",
        "conflict"
      ],
      "facts": {
        "operation": "op_A",
        "attempt": "a3",
        "payload": "P2",
        "effect_committed": true,
        "result_recorded": true,
        "duplicate_effect": false,
        "response": "conflict",
        "authorized": true
      },
      "camera": [
        535,
        140
      ],
      "motion": {
        "label": "conflict",
        "route": [
          [
            370,
            154
          ],
          [
            450,
            154
          ],
          [
            450,
            210
          ],
          [
            535,
            210
          ]
        ],
        "role": "fault",
        "lock_camera": false
      }
    }
  ],
  "name": "original-v3-idempotency-flow"
};
const NS='http://www.w3.org/2000/svg';
window.MLSysFigureLessons=window.MLSysFigureLessons||{};
window.MLSysFigureLessons[spec.name]=async(root,context)=>{
 if(!context.verifiedArtwork||!window.MLSysOriginalPlayer)throw Error('Verified original artwork and shared player required');
 const svg=context.verifiedArtwork.cloneNode(true),vb=svg.viewBox.baseVal;
 if(svg.getAttribute('viewBox')!==spec.viewBox)throw Error('Unreviewed original diagram geometry');
 root.classList.add(spec.name);svg.setAttribute('role','img');svg.setAttribute('aria-label',spec.question+' Original chapter diagram with synchronized state.');
 const node=(tag,a={},text)=>{const n=document.createElementNS(NS,tag);Object.entries(a).forEach(([k,v])=>n.setAttribute(k,String(v)));if(text!==undefined)n.textContent=text;return n;};
 const textNodes=[...svg.querySelectorAll('text')],bindings={};
 for(const [key,b]of Object.entries(spec.slots)){const matches=textNodes.filter(n=>Number(n.getAttribute('x'))===b.x&&Number(n.getAttribute('y'))===b.y);if(matches.length!==1)throw Error('Original slot changed: '+key);bindings[key]={node:matches[0],original:matches[0].innerHTML,font:matches[0].getAttribute('font-size')};}
 const circles=[...svg.querySelectorAll('circle')].map(n=>({n,fill:n.getAttribute('fill'),stroke:n.getAttribute('stroke'),'stroke-width':n.getAttribute('stroke-width')}));
 const overlay=node('g',{class:'v3-original-overlay'});svg.append(overlay);
 const markerLayer=node('g');overlay.append(markerLayer);
 const packet=node('g',{class:'v3-original-packet',visibility:'hidden'});packet.append(node('rect',{x:-24,y:-10,width:48,height:20,rx:3}),node('text',{'text-anchor':'middle',y:4}));overlay.append(packet);
 const canvas=document.createElement('div');canvas.className='v3-original-canvas';canvas.tabIndex=0;canvas.setAttribute('aria-label','Original diagram. Swipe horizontally or vertically to inspect; the view follows the active object.');canvas.append(svg);
 const cue=document.createElement('p');cue.className='v3-original-cue';cue.textContent='Swipe the original diagram to inspect. The view follows the active object; Expand keeps the same state.';
 const readout=document.createElement('div');readout.className='v3-original-readout';readout.setAttribute('aria-label','Current systems state');
 const values=spec.fields.map(label=>{const row=document.createElement('div'),name=document.createElement('span'),value=document.createElement('strong');name.textContent=label;row.append(name,value);readout.append(row);return value;});
 let variant=spec.variants?.[0].id||'',player,active=[vb.width/2,vb.height/2];
 const frames=()=>spec.variants?spec.variants.find(v=>v.id===variant).states:spec.states;
 const steps=frames().map(s=>({label:s.label,description:s.description}));
 let choice;
 if(spec.variants){choice=document.createElement('label');choice.className='v3-original-choice';choice.textContent=spec.variantLabel+' ';const select=document.createElement('select');select.setAttribute('aria-label',spec.variantLabel);spec.variants.forEach(v=>{const o=document.createElement('option');o.value=v.id;o.textContent=v.label;select.append(o);});select.addEventListener('change',()=>{variant=select.value;frames().forEach((s,i)=>Object.assign(steps[i],{label:s.label,description:s.description}));player.go(0);});choice.append(select);}
 function follow(x,y){active=[x,y];const scale=svg.getBoundingClientRect().width/vb.width;if(canvas.scrollWidth>canvas.clientWidth)canvas.scrollLeft=Math.max(0,x*scale-canvas.clientWidth/2);if(canvas.scrollHeight>canvas.clientHeight)canvas.scrollTop=Math.max(0,y*scale-canvas.clientHeight/2);}
 function render(i,{stage,details}){
  if(!canvas.isConnected){if(choice)stage.append(choice);stage.append(cue,canvas,readout);}
  const frame=frames()[i];packet.setAttribute('visibility','hidden');markerLayer.replaceChildren();
  for(const [key,b]of Object.entries(bindings)){b.node.innerHTML=b.original;if(b.font)b.node.setAttribute('font-size',b.font);b.node.removeAttribute('opacity');const value=Object.assign({},spec.defaults,frame.text)[key];if(value!==undefined)b.node.textContent=value;if(spec.slots[key].font)b.node.setAttribute('font-size',spec.slots[key].font);b.node.removeAttribute('textLength');if(spec.slots[key].width&&b.node.getComputedTextLength()>spec.slots[key].width)b.node.setAttribute('textLength',spec.slots[key].width);b.node.setAttribute('lengthAdjust','spacingAndGlyphs');}
  for(const c of circles)for(const k of ['fill','stroke','stroke-width']){if(c[k]!==null)c.n.setAttribute(k,c[k]);}
  for(const p of frame.circles||[]){const c=circles.find(c=>Number(c.n.getAttribute('cx'))===p.x&&Number(c.n.getAttribute('cy'))===p.y&&Number(c.n.getAttribute('r'))===(p.r||8));if(!c)throw Error('Original graph node changed');for(const k of ['fill','stroke','stroke-width'])if(p[k])c.n.setAttribute(k,p[k]);}
  for(const m of frame.marks||[]){markerLayer.append(node('circle',{cx:m.x,cy:m.y,r:m.r||12,fill:'none',stroke:m.color||'#0284c7','stroke-width':2}));}
  values.forEach((v,j)=>v.textContent=frame.values[j]);root.dataset.originalStep=String(i);root.dataset.originalFacts=JSON.stringify(frame.facts);root.dataset.originalVariant=variant;
  follow(...frame.camera);
  if(!details.firstChild){const p=document.createElement('p');p.textContent=spec.guard;details.append(p);}
 }
 function transition({from,to,onFrame,animate}){
  if(to!==from+1)return;const frame=frames()[to],motion=frame.motion;
  if(!motion){for(const key of frame.fade||[]){const n=bindings[key]?.node;if(n)animate(n,[{opacity:1},{opacity:.15}],{});}return;}
  const route=motion.route,lengths=route.slice(1).map((p,i)=>Math.hypot(p[0]-route[i][0],p[1]-route[i][1])),total=lengths.reduce((a,b)=>a+b,0);
  const width=Math.max(28,Math.min(112,motion.label.length*6+12));packet.querySelector('rect').setAttribute('x',-width/2);packet.querySelector('rect').setAttribute('width',width);packet.querySelector('text').textContent=motion.label;packet.dataset.role=motion.role||'state';packet.setAttribute('visibility','visible');
  onFrame(p=>{let d=total*p,j=0;while(j<lengths.length-1&&d>lengths[j]){d-=lengths[j];j++;}const t=lengths[j]?d/lengths[j]:0,a=route[j],b=route[j+1],x=a[0]+(b[0]-a[0])*t,y=a[1]+(b[1]-a[1])*t;packet.setAttribute('transform',`translate(${x} ${y})`);follow(...(motion.lock_camera?frame.camera:[x,y]));if(p>=1)packet.setAttribute('visibility','hidden');});
 }
 player=window.MLSysOriginalPlayer.create({root,outer:context.outer,title:spec.question,scope:spec.scope,steps,render,transition});
 const resize=new ResizeObserver(()=>follow(...active));resize.observe(canvas);const destroy=player.destroy;player.destroy=()=>{resize.disconnect();destroy();};return player;
};
})();

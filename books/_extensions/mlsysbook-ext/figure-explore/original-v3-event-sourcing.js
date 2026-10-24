/* Original chapter artwork with inspectable, endpoint-driven systems state. */
(() => {
'use strict';
const spec = {
  "viewBox": "0 0 760 370",
  "asset": "vol3-event-sourcing.svg",
  "question": "What survives a crash, and what must remain unknown?",
  "scope": "The original recorded event tape for test_r18. Recovery folds preserved payloads in log order. There is no recorded ToolResult. Token movement represents logical event application, not measured storage transfer.",
  "purpose": "Make the durable log survive while volatile state disappears, then rebuild admission, reservation, and unresolved intent without replaying external effects.",
  "guard": "The log is authoritative only within its declared durability and payload-retention contract. Missing outcomes remain unknown. Reducer replay is not a model call or a tool invocation; crash recovery does not release an unresolved reservation.",
  "slots": {
    "header": {
      "x": 630,
      "y": 78,
      "width": 160,
      "font": 13
    },
    "admit": {
      "x": 630,
      "y": 108,
      "width": 150,
      "font": 12
    },
    "reserved": {
      "x": 630,
      "y": 132,
      "width": 150,
      "font": 12
    },
    "result": {
      "x": 630,
      "y": 156,
      "width": 150,
      "font": 12
    },
    "reducer": {
      "x": 380,
      "y": 120,
      "width": 160,
      "font": 12
    }
  },
  "defaults": {},
  "fields": [
    "Durable log",
    "Projection",
    "Reservation",
    "Test outcome",
    "Replay side effects"
  ],
  "states": [
    {
      "label": "Retain the durable event tape",
      "description": "The append-only tape ends at the recorded test_r18 intent. The live projection holds its admission and reservation, but no result was recorded.",
      "text": {
        "header": "Live record before crash",
        "admit": "admission held",
        "reserved": "reservation held",
        "result": "test_r18 unresolved",
        "reducer": "apply(R, e)"
      },
      "values": [
        "Durable prefix unchanged",
        "Live",
        "Held in log and projection",
        "No recorded ToolResult",
        "No model/tool re-execution"
      ],
      "facts": {
        "crashed": false,
        "folded": 0,
        "authorized": true,
        "projection_reserved": true,
        "durable_reservation": true,
        "projection_intent": true,
        "durable_intent": true,
        "result_recorded": false,
        "model_reexecuted": false,
        "tool_reexecuted": false,
        "log_intact": true
      },
      "camera": [
        630,
        125
      ],
      "fade": []
    },
    {
      "label": "Lose the volatile projection",
      "description": "A crash removes the volatile projection. When recovery begins, it must reconstruct from the durable tape; this action does not execute the model or test again.",
      "text": {
        "header": "Projection lost on crash",
        "admit": "volatile record lost",
        "reserved": "rebuild from R0",
        "result": "outcome still unknown",
        "reducer": "apply(R, e)"
      },
      "values": [
        "Durable prefix unchanged",
        "Lost",
        "Held in log; projection not yet folded",
        "No recorded ToolResult",
        "No model/tool re-execution"
      ],
      "facts": {
        "crashed": true,
        "folded": 0,
        "authorized": false,
        "projection_reserved": false,
        "durable_reservation": true,
        "projection_intent": false,
        "durable_intent": true,
        "result_recorded": false,
        "model_reexecuted": false,
        "tool_reexecuted": false,
        "log_intact": true
      },
      "camera": [
        630,
        125
      ],
      "fade": [
        "admit",
        "reserved",
        "result"
      ]
    },
    {
      "label": "Fold TaskStarted",
      "description": "Feed the recorded TaskStarted event through the deterministic reducer. When the fold finishes, task identity, grants, and limits are reconstructed.",
      "text": {
        "header": "Reconstructed record",
        "admit": "task reconstructed",
        "reserved": "no admitted reservation",
        "result": "no tool result replayed",
        "reducer": "apply(R, e1)"
      },
      "values": [
        "Durable prefix unchanged",
        "Recovered through e1",
        "Held in log; projection not yet folded",
        "No recorded ToolResult",
        "No model/tool re-execution"
      ],
      "facts": {
        "crashed": true,
        "folded": 1,
        "authorized": false,
        "projection_reserved": false,
        "durable_reservation": true,
        "projection_intent": false,
        "durable_intent": true,
        "result_recorded": false,
        "model_reexecuted": false,
        "tool_reexecuted": false,
        "log_intact": true
      },
      "camera": [
        630,
        125
      ],
      "fade": [],
      "motion": {
        "label": "e1",
        "route": [
          [
            95,
            292
          ],
          [
            95,
            235
          ],
          [
            380,
            235
          ],
          [
            380,
            172
          ],
          [
            380,
            115
          ],
          [
            538,
            115
          ],
          [
            630,
            115
          ]
        ],
        "role": "state",
        "lock_camera": false
      }
    },
    {
      "label": "Fold ContextBuilt",
      "description": "Fold the recorded ContextBuilt payload. When this action finishes, the delivered context is part of the reconstructed history; it is not assembled by a new model call.",
      "text": {
        "header": "Reconstructed record",
        "admit": "context reconstructed",
        "reserved": "no admitted reservation",
        "result": "no tool result replayed",
        "reducer": "apply(R, e2)"
      },
      "values": [
        "Durable prefix unchanged",
        "Recovered through e2",
        "Held in log; projection not yet folded",
        "No recorded ToolResult",
        "No model/tool re-execution"
      ],
      "facts": {
        "crashed": true,
        "folded": 2,
        "authorized": false,
        "projection_reserved": false,
        "durable_reservation": true,
        "projection_intent": false,
        "durable_intent": true,
        "result_recorded": false,
        "model_reexecuted": false,
        "tool_reexecuted": false,
        "log_intact": true
      },
      "camera": [
        630,
        125
      ],
      "fade": [],
      "motion": {
        "label": "e2",
        "route": [
          [
            205,
            292
          ],
          [
            205,
            235
          ],
          [
            380,
            235
          ],
          [
            380,
            172
          ],
          [
            380,
            115
          ],
          [
            538,
            115
          ],
          [
            630,
            115
          ]
        ],
        "role": "state",
        "lock_camera": false
      }
    },
    {
      "label": "Fold ModelReplied",
      "description": "Fold the recorded ModelReplied event. When the action finishes, the old proposal is restored from its recorded payload, not sampled again.",
      "text": {
        "header": "Reconstructed record",
        "admit": "proposal recorded",
        "reserved": "no admitted reservation",
        "result": "no tool result replayed",
        "reducer": "apply(R, e3)"
      },
      "values": [
        "Durable prefix unchanged",
        "Recovered through e3",
        "Held in log; projection not yet folded",
        "No recorded ToolResult",
        "No model/tool re-execution"
      ],
      "facts": {
        "crashed": true,
        "folded": 3,
        "authorized": false,
        "projection_reserved": false,
        "durable_reservation": true,
        "projection_intent": false,
        "durable_intent": true,
        "result_recorded": false,
        "model_reexecuted": false,
        "tool_reexecuted": false,
        "log_intact": true
      },
      "camera": [
        630,
        125
      ],
      "fade": [],
      "motion": {
        "label": "e3",
        "route": [
          [
            315,
            292
          ],
          [
            315,
            235
          ],
          [
            380,
            235
          ],
          [
            380,
            172
          ],
          [
            380,
            115
          ],
          [
            538,
            115
          ],
          [
            630,
            115
          ]
        ],
        "role": "state",
        "lock_camera": false
      }
    },
    {
      "label": "Fold authorization and reservation",
      "description": "Fold the recorded authorization and budget reservation. When this action finishes, both admission and its held reservation are represented; recovery does not return funds merely because the worker crashed.",
      "text": {
        "header": "Reconstructed record",
        "admit": "admission held",
        "reserved": "reservation held",
        "result": "no tool result replayed",
        "reducer": "apply(R, e4)"
      },
      "values": [
        "Durable prefix unchanged",
        "Recovered through e4",
        "Held in log and projection",
        "No recorded ToolResult",
        "No model/tool re-execution"
      ],
      "facts": {
        "crashed": true,
        "folded": 4,
        "authorized": true,
        "projection_reserved": true,
        "durable_reservation": true,
        "projection_intent": false,
        "durable_intent": true,
        "result_recorded": false,
        "model_reexecuted": false,
        "tool_reexecuted": false,
        "log_intact": true
      },
      "camera": [
        630,
        125
      ],
      "fade": [],
      "motion": {
        "label": "e4",
        "route": [
          [
            425,
            292
          ],
          [
            425,
            235
          ],
          [
            380,
            235
          ],
          [
            380,
            172
          ],
          [
            380,
            115
          ],
          [
            538,
            115
          ],
          [
            630,
            115
          ]
        ],
        "role": "state",
        "lock_camera": false
      }
    },
    {
      "label": "Fold the unresolved tool intent",
      "description": "Fold ToolIntent test_r18. When this action finishes, the reconstructed operation remains unresolved because ToolResult is absent. Neither a success nor a failure is invented, and no tool is re-executed.",
      "text": {
        "header": "Reconstructed record",
        "admit": "admission held",
        "reserved": "reservation held",
        "result": "test_r18 unresolved",
        "reducer": "apply(R, ej)"
      },
      "values": [
        "Durable prefix unchanged",
        "Recovered through ej",
        "Held in log and projection",
        "No recorded ToolResult",
        "No model/tool re-execution"
      ],
      "facts": {
        "crashed": true,
        "folded": 5,
        "authorized": true,
        "projection_reserved": true,
        "durable_reservation": true,
        "projection_intent": true,
        "durable_intent": true,
        "result_recorded": false,
        "model_reexecuted": false,
        "tool_reexecuted": false,
        "log_intact": true
      },
      "camera": [
        630,
        125
      ],
      "fade": [],
      "motion": {
        "label": "ej",
        "route": [
          [
            535,
            292
          ],
          [
            535,
            235
          ],
          [
            380,
            235
          ],
          [
            380,
            172
          ],
          [
            380,
            115
          ],
          [
            538,
            115
          ],
          [
            630,
            115
          ]
        ],
        "role": "state",
        "lock_camera": false
      }
    }
  ],
  "name": "original-v3-event-sourcing"
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

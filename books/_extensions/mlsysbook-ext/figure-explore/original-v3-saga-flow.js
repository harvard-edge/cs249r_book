/* Original chapter artwork with inspectable, endpoint-driven systems state. */
(() => {
'use strict';
const spec = {
  "viewBox": "0 0 760 480",
  "asset": "vol3-saga-flow.svg",
  "question": "Who owns cleanup when an external result is unknown?",
  "intro": "Trace 10 saga lifecycle steps from pre-dispatch obligation registration through lost provisioning replies and DAG preconditions to confirmed environment release.",
  "scope": "Provisioning and semantic compensation saga for temporary integration-test environment env_r18_01 across the Forward operation panel (top) and Registered cleanup DAG panel (bottom). The provisioning reply for provision_test_env_r18 is lost, correlated evidence later binds target env_r18_01, and release_test_env_r18 waits for two DAG predecessors (test consumer shutdown and external artifact preservation) while local scratch removal remains independent.",
  "purpose": "Keep one cleanup obligation alive through registration, unknown provisioning, target binding, gated release, and confirmed closure.",
  "guard": "Register cleanup before external provisioning. Unknown outcomes retain ownership. Binding requires correlated evidence; closing requires confirmed release. Cancellation or test completion and artifact preservation are separate conditions. Saga compensation does not atomically undo the world.",
  "slots": {
    "closedhead": {
      "x": 625,
      "y": 299,
      "width": 175,
      "font": 12
    },
    "retainedhead": {
      "x": 625,
      "y": 359,
      "width": 175,
      "font": 12
    },
    "unknownhead": {
      "x": 380,
      "y": 157,
      "width": 210,
      "font": 12
    },
    "reg": {
      "x": 135,
      "y": 94,
      "width": 175,
      "font": 11
    },
    "dispatch": {
      "x": 380,
      "y": 94,
      "width": 175,
      "font": 11
    },
    "bind": {
      "x": 625,
      "y": 94,
      "width": 175,
      "font": 11
    },
    "unknown": {
      "x": 380,
      "y": 175,
      "width": 210,
      "font": 11
    },
    "test": {
      "x": 130,
      "y": 317,
      "width": 160,
      "font": 11
    },
    "artifact": {
      "x": 130,
      "y": 377,
      "width": 160,
      "font": 11
    },
    "release": {
      "x": 380,
      "y": 347,
      "width": 145,
      "font": 11
    },
    "closed": {
      "x": 625,
      "y": 317,
      "width": 175,
      "font": 11
    },
    "retained": {
      "x": 625,
      "y": 377,
      "width": 175,
      "font": 11
    }
  },
  "defaults": {},
  "fields": [
    "Cleanup registration",
    "Bound target",
    "Test consumers",
    "External artifacts",
    "Cleanup ownership"
  ],
  "states": [
    {
      "label": "Keep cleanup ownership explicit",
      "description": "At the start of the 10-step saga, the Forward operation lane (top) and Registered cleanup lane (bottom) separate cleanup obligation ownership from the concrete resource identifier, which is unknown until provisioning completes.",
      "text": {
        "closedhead": "Closure gate",
        "retainedhead": "Retained, owned",
        "unknownhead": "Outcome branch",
        "reg": "not yet registered",
        "dispatch": "not dispatched",
        "bind": "target not yet known",
        "unknown": "no dispatch yet",
        "test": "not established stopped",
        "artifact": "preservation not established",
        "release": "preconditions not yet met",
        "closed": "not closed",
        "retained": "no obligation registered"
      },
      "values": [
        "Not registered",
        "Unbound (no resource ID yet)",
        "Not confirmed stopped",
        "Not established",
        "Not owned (pre-registration)"
      ],
      "facts": {
        "registered": false,
        "provision_dispatched": false,
        "bound": false,
        "test_stopped": false,
        "artifacts_preserved": false,
        "release_dispatched": false,
        "release_confirmed": false,
        "closed": false,
        "owned": false,
        "scratch_removal_sufficient": false
      },
      "camera": [
        135,
        110
      ]
    },
    {
      "label": "Register the cleanup obligation",
      "description": "In the top-left purple box (Register obligation), durably record the compensating template release_test_env_r18 before sending any network request. The solid purple link registers the obligation in the bottom cleanup lane with target unbound.",
      "text": {
        "closedhead": "Closure gate",
        "retainedhead": "Retained, owned",
        "unknownhead": "Outcome branch",
        "reg": "release_env: registered, unbound",
        "dispatch": "not dispatched",
        "bind": "target not yet known",
        "unknown": "no dispatch yet",
        "test": "not established stopped",
        "artifact": "preservation not established",
        "release": "preconditions not yet met",
        "closed": "not closed",
        "retained": "owned; target unbound"
      },
      "values": [
        "Registered before dispatch",
        "Unbound (awaiting creation evidence)",
        "Not confirmed stopped",
        "Not established",
        "Owned (pre-dispatch obligation active)"
      ],
      "facts": {
        "registered": true,
        "provision_dispatched": false,
        "bound": false,
        "test_stopped": false,
        "artifacts_preserved": false,
        "release_dispatched": false,
        "release_confirmed": false,
        "closed": false,
        "owned": true,
        "scratch_removal_sufficient": false
      },
      "camera": [
        135,
        110
      ],
      "motion": {
        "label": "register",
        "route": [
          [
            135,
            75
          ],
          [
            135,
            108
          ],
          [
            135,
            222
          ],
          [
            350,
            222
          ],
          [
            350,
            303
          ]
        ],
        "role": "state",
        "lock_camera": false
      }
    },
    {
      "label": "Dispatch provisioning",
      "description": "Once the cleanup obligation is durably registered, the runtime dispatches provision_test_env_r18 (top-center amber box). Because a remote environment may now be allocating and accruing charges, cleanup ownership remains active.",
      "text": {
        "closedhead": "Closure gate",
        "retainedhead": "Retained, owned",
        "unknownhead": "Awaiting evidence",
        "reg": "release_env: registered, unbound",
        "dispatch": "provision_test_env_r18",
        "bind": "target not yet known",
        "unknown": "awaiting provisioning outcome",
        "test": "not established stopped",
        "artifact": "preservation not established",
        "release": "preconditions not yet met",
        "closed": "not closed",
        "retained": "owned; target unbound"
      },
      "values": [
        "Registered",
        "Unbound (provisioning in flight)",
        "Not confirmed stopped",
        "Not established",
        "Owned (provisioning in flight)"
      ],
      "facts": {
        "registered": true,
        "provision_dispatched": true,
        "bound": false,
        "test_stopped": false,
        "artifacts_preserved": false,
        "release_dispatched": false,
        "release_confirmed": false,
        "closed": false,
        "owned": true,
        "scratch_removal_sufficient": false
      },
      "camera": [
        380,
        150
      ],
      "motion": {
        "label": "provision",
        "route": [
          [
            230,
            80
          ],
          [
            283,
            80
          ],
          [
            380,
            80
          ]
        ],
        "role": "tool",
        "lock_camera": false
      }
    },
    {
      "label": "Lose the provisioning reply",
      "description": "The provisioning call times out along the red reply lost arrow into the dotted red box (Outcome unknown). Because the remote environment may still exist, the runtime keeps the unbound obligation open and owned rather than assuming creation failed.",
      "text": {
        "closedhead": "Closure gate",
        "retainedhead": "Retained, owned",
        "unknownhead": "Outcome unknown",
        "reg": "release_env: registered, unbound",
        "dispatch": "provision_test_env_r18",
        "bind": "target not yet known",
        "unknown": "reply lost: outcome unknown",
        "test": "not established stopped",
        "artifact": "preservation not established",
        "release": "preconditions not yet met",
        "closed": "not closed",
        "retained": "owned; target unbound"
      },
      "values": [
        "Registered",
        "Unbound (reply lost; outcome unknown)",
        "Not confirmed stopped",
        "Not established",
        "Owned (open in-doubt obligation)"
      ],
      "facts": {
        "registered": true,
        "provision_dispatched": true,
        "bound": false,
        "test_stopped": false,
        "artifacts_preserved": false,
        "release_dispatched": false,
        "release_confirmed": false,
        "closed": false,
        "owned": true,
        "scratch_removal_sufficient": false
      },
      "camera": [
        380,
        150
      ],
      "motion": {
        "label": "reply lost",
        "route": [
          [
            380,
            108
          ],
          [
            380,
            157
          ]
        ],
        "role": "fault",
        "lock_camera": false
      }
    },
    {
      "label": "Bind correlated resource evidence",
      "description": "Authenticated reconciliation evidence arrives at the top-right purple box (Bind on evidence), identifying resource handle env_r18_01. The dashed purple link binds env_r18_01 as the concrete target of release_test_env_r18 in the cleanup lane.",
      "text": {
        "closedhead": "Closure gate",
        "retainedhead": "Retained, owned",
        "unknownhead": "Evidence correlated",
        "reg": "release_env: registered, unbound",
        "dispatch": "provision_test_env_r18",
        "bind": "bound: env_r18_01",
        "unknown": "correlated evidence received",
        "test": "not established stopped",
        "artifact": "preservation not established",
        "release": "preconditions not yet met",
        "closed": "not closed",
        "retained": "owned; env_r18_01"
      },
      "values": [
        "Registered",
        "env_r18_01 (bound via correlated evidence)",
        "Not confirmed stopped (1 / 2 DAG edges)",
        "Not established (0 / 2 DAG edges)",
        "Owned (awaiting cleanup DAG predecessors)"
      ],
      "facts": {
        "registered": true,
        "provision_dispatched": true,
        "bound": true,
        "test_stopped": false,
        "artifacts_preserved": false,
        "release_dispatched": false,
        "release_confirmed": false,
        "closed": false,
        "owned": true,
        "scratch_removal_sufficient": false
      },
      "camera": [
        625,
        140
      ],
      "motion": {
        "label": "env_r18_01",
        "route": [
          [
            625,
            94
          ],
          [
            625,
            222
          ],
          [
            410,
            222
          ],
          [
            410,
            303
          ]
        ],
        "role": "memory",
        "lock_camera": false
      }
    },
    {
      "label": "Stop the test consumers",
      "description": "In the bottom-left cleanup DAG, satisfy the first predecessor node (Test ended or cancel confirmed) so no active integration test still depends on env_r18_01. Release remains blocked until the second predecessor (Artifacts preserved) also settles.",
      "text": {
        "closedhead": "Closure gate",
        "retainedhead": "Retained, owned",
        "unknownhead": "Evidence correlated",
        "reg": "release_env: registered, unbound",
        "dispatch": "provision_test_env_r18",
        "bind": "bound: env_r18_01",
        "unknown": "correlated evidence received",
        "test": "stopped; no remaining consumer",
        "artifact": "preservation not established",
        "release": "preconditions not yet met",
        "closed": "not closed",
        "retained": "owned; env_r18_01"
      },
      "values": [
        "Registered",
        "env_r18_01",
        "Stopped (predecessor 1 / 2 satisfied)",
        "Not established (blocks release)",
        "Owned (waiting for artifact export)"
      ],
      "facts": {
        "registered": true,
        "provision_dispatched": true,
        "bound": true,
        "test_stopped": true,
        "artifacts_preserved": false,
        "release_dispatched": false,
        "release_confirmed": false,
        "closed": false,
        "owned": true,
        "scratch_removal_sufficient": false
      },
      "camera": [
        130,
        340
      ],
      "motion": {
        "label": "stopped",
        "route": [
          [
            130,
            303
          ],
          [
            220,
            303
          ],
          [
            270,
            303
          ],
          [
            270,
            323
          ],
          [
            298,
            323
          ]
        ],
        "role": "memory",
        "lock_camera": false
      }
    },
    {
      "label": "Preserve external test artifacts",
      "description": "Export diagnostic logs to durable storage outside the sandbox, satisfying the second DAG predecessor (Artifacts preserved). Notice that Scratch removal below has no dependency edge into Release and cannot substitute for either precondition.",
      "text": {
        "closedhead": "Closure gate",
        "retainedhead": "Retained, owned",
        "unknownhead": "Evidence correlated",
        "reg": "release_env: registered, unbound",
        "dispatch": "provision_test_env_r18",
        "bind": "bound: env_r18_01",
        "unknown": "correlated evidence received",
        "test": "stopped; no remaining consumer",
        "artifact": "external artifacts preserved",
        "release": "preconditions not yet met",
        "closed": "not closed",
        "retained": "owned; env_r18_01"
      },
      "values": [
        "Registered",
        "env_r18_01",
        "Stopped (predecessor 1 / 2 satisfied)",
        "Preserved (predecessor 2 / 2 satisfied)",
        "Owned (both DAG predecessors met)"
      ],
      "facts": {
        "registered": true,
        "provision_dispatched": true,
        "bound": true,
        "test_stopped": true,
        "artifacts_preserved": true,
        "release_dispatched": false,
        "release_confirmed": false,
        "closed": false,
        "owned": true,
        "scratch_removal_sufficient": false
      },
      "camera": [
        130,
        340
      ],
      "motion": {
        "label": "preserved",
        "route": [
          [
            130,
            363
          ],
          [
            220,
            363
          ],
          [
            270,
            363
          ],
          [
            270,
            343
          ],
          [
            298,
            343
          ]
        ],
        "role": "memory",
        "lock_camera": false
      }
    },
    {
      "label": "Admit resource release",
      "description": "With target env_r18_01 bound and both DAG predecessors settled, the coordinator admits and dispatches release_test_env_r18 (bottom-center amber box). Dispatching the compensating step does not close the obligation until postcondition proof arrives.",
      "text": {
        "closedhead": "Closure gate",
        "retainedhead": "Retained, owned",
        "unknownhead": "Evidence correlated",
        "reg": "release_env: registered, unbound",
        "dispatch": "provision_test_env_r18",
        "bind": "bound: env_r18_01",
        "unknown": "correlated evidence received",
        "test": "stopped; no remaining consumer",
        "artifact": "external artifacts preserved",
        "release": "release sent; awaiting proof",
        "closed": "not closed",
        "retained": "owned; env_r18_01"
      },
      "values": [
        "Registered",
        "env_r18_01",
        "Stopped",
        "Preserved",
        "Owned (release_test_env_r18 in flight)"
      ],
      "facts": {
        "registered": true,
        "provision_dispatched": true,
        "bound": true,
        "test_stopped": true,
        "artifacts_preserved": true,
        "release_dispatched": true,
        "release_confirmed": false,
        "closed": false,
        "owned": true,
        "scratch_removal_sufficient": false
      },
      "camera": [
        380,
        340
      ],
      "motion": {
        "label": "release",
        "route": [
          [
            350,
            320
          ],
          [
            380,
            328
          ],
          [
            475,
            328
          ]
        ],
        "role": "tool",
        "lock_camera": false
      }
    },
    {
      "label": "Retain an unconfirmed release",
      "description": "If the release call times out or returns only transport acceptance, execution follows the lower red arrow into the dotted red box (Retained, owned · release unconfirmed), keeping env_r18_01 on record for retry or escalation.",
      "text": {
        "closedhead": "Closure gate",
        "retainedhead": "Retained, owned",
        "unknownhead": "Evidence correlated",
        "reg": "release_env: registered, unbound",
        "dispatch": "provision_test_env_r18",
        "bind": "bound: env_r18_01",
        "unknown": "correlated evidence received",
        "test": "stopped; no remaining consumer",
        "artifact": "external artifacts preserved",
        "release": "release sent; awaiting proof",
        "closed": "not closed",
        "retained": "owned; env_r18_01"
      },
      "values": [
        "Registered",
        "env_r18_01",
        "Stopped",
        "Preserved",
        "Owned (release unconfirmed; retained)"
      ],
      "facts": {
        "registered": true,
        "provision_dispatched": true,
        "bound": true,
        "test_stopped": true,
        "artifacts_preserved": true,
        "release_dispatched": true,
        "release_confirmed": false,
        "closed": false,
        "owned": true,
        "scratch_removal_sufficient": false
      },
      "camera": [
        625,
        340
      ],
      "motion": {
        "label": "unknown",
        "route": [
          [
            475,
            328
          ],
          [
            520,
            328
          ],
          [
            520,
            363
          ],
          [
            625,
            363
          ]
        ],
        "role": "fault",
        "lock_camera": false
      }
    },
    {
      "label": "Close only after confirmed release",
      "description": "When authenticated endpoint evidence confirms that env_r18_01 is deallocated, execution follows the upper green arrow to Obligation closed (postcondition shown), establishing semantic equivalence on the declared resource predicates while leaving billing settlement separate.",
      "text": {
        "closedhead": "Obligation closed",
        "retainedhead": "Ownership record",
        "unknownhead": "Evidence correlated",
        "reg": "release_env: registered, unbound",
        "dispatch": "provision_test_env_r18",
        "bind": "bound: env_r18_01",
        "unknown": "correlated evidence received",
        "test": "stopped; no remaining consumer",
        "artifact": "external artifacts preserved",
        "release": "release confirmed",
        "closed": "obligation closed",
        "retained": "closed after evidence"
      },
      "values": [
        "Registered",
        "env_r18_01",
        "Stopped",
        "Preserved",
        "Closed on postcondition confirmation"
      ],
      "facts": {
        "registered": true,
        "provision_dispatched": true,
        "bound": true,
        "test_stopped": true,
        "artifacts_preserved": true,
        "release_dispatched": true,
        "release_confirmed": true,
        "closed": true,
        "owned": false,
        "scratch_removal_sufficient": false
      },
      "camera": [
        625,
        340
      ],
      "motion": {
        "label": "confirmed",
        "route": [
          [
            475,
            328
          ],
          [
            520,
            328
          ],
          [
            520,
            303
          ],
          [
            625,
            303
          ]
        ],
        "role": "memory",
        "lock_camera": false
      }
    }
  ],
  "name": "original-v3-saga-flow"
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
  if(!canvas.isConnected){stage.append(cue,canvas);details.append(readout);}
  const frame=frames()[i];packet.setAttribute('visibility','hidden');markerLayer.replaceChildren();
  for(const [key,b]of Object.entries(bindings)){b.node.innerHTML=b.original;if(b.font)b.node.setAttribute('font-size',b.font);b.node.removeAttribute('opacity');const value=Object.assign({},spec.defaults,frame.text)[key];if(value!==undefined)b.node.textContent=value;if(spec.slots[key].font)b.node.setAttribute('font-size',spec.slots[key].font);b.node.removeAttribute('textLength');if(spec.slots[key].width&&b.node.getComputedTextLength()>spec.slots[key].width)b.node.setAttribute('textLength',spec.slots[key].width);b.node.setAttribute('lengthAdjust','spacingAndGlyphs');}
  for(const c of circles)for(const k of ['fill','stroke','stroke-width']){if(c[k]!==null)c.n.setAttribute(k,c[k]);}
  for(const p of frame.circles||[]){const c=circles.find(c=>Number(c.n.getAttribute('cx'))===p.x&&Number(c.n.getAttribute('cy'))===p.y&&Number(c.n.getAttribute('r'))===(p.r||8));if(!c)throw Error('Original graph node changed');for(const k of ['fill','stroke','stroke-width'])if(p[k])c.n.setAttribute(k,p[k]);}
  for(const m of frame.marks||[]){markerLayer.append(node('circle',{cx:m.x,cy:m.y,r:m.r||12,fill:'none',stroke:m.color||'#0284c7','stroke-width':2}));}
  values.forEach((v,j)=>v.textContent=frame.values[j]);root.dataset.originalStep=String(i);root.dataset.originalFacts=JSON.stringify(frame.facts);root.dataset.originalVariant=variant;
  follow(...frame.camera);
  if(!details.querySelector('.v3-original-guard')){const p=document.createElement('p');p.className='v3-original-guard';p.textContent=spec.guard;details.append(p);}
 }
 function transition({from,to,onFrame,animate}){
  if(to!==from+1)return;const frame=frames()[to],motion=frame.motion;
  if(!motion){for(const key of frame.fade||[]){const n=bindings[key]?.node;if(n)animate(n,[{opacity:1},{opacity:.15}],{});}return;}
  const route=motion.route,lengths=route.slice(1).map((p,i)=>Math.hypot(p[0]-route[i][0],p[1]-route[i][1])),total=lengths.reduce((a,b)=>a+b,0);
  const width=Math.max(28,Math.min(112,motion.label.length*6+12));packet.querySelector('rect').setAttribute('x',-width/2);packet.querySelector('rect').setAttribute('width',width);packet.querySelector('text').textContent=motion.label;packet.dataset.role=motion.role||'state';packet.setAttribute('visibility','visible');
  onFrame(p=>{let d=total*p,j=0;while(j<lengths.length-1&&d>lengths[j]){d-=lengths[j];j++;}const t=lengths[j]?d/lengths[j]:0,a=route[j],b=route[j+1],x=a[0]+(b[0]-a[0])*t,y=a[1]+(b[1]-a[1])*t;packet.setAttribute('transform',`translate(${x} ${y})`);follow(...(motion.lock_camera?frame.camera:[x,y]));if(p>=1)packet.setAttribute('visibility','hidden');});
 }
 player=window.MLSysOriginalPlayer.create({root,outer:context.outer,title:spec.question,intro:spec.intro,layout:'compact',viewControls:choice,scope:spec.scope,steps,render,transition});
 const resize=new ResizeObserver(()=>follow(...active));resize.observe(canvas);const destroy=player.destroy;player.destroy=()=>{resize.disconnect();destroy();};return player;
};
})();

/* Original chapter artwork with inspectable, endpoint-driven systems state. */
(() => {
'use strict';
const spec = {
  "viewBox": "0 0 740 440",
  "asset": "fig-vol3-occ-worktrees.svg",
  "question": "Why can a checked private change still fail publication?",
  "scope": "Illustrative identities h0/h1/h2 and c1/c2/c3 on the original worktree, dependency, candidate, and publication diagram. Each head is a revision identity, not an elapsed-time measurement. Candidate checks and captured target identity travel together.",
  "purpose": "Make semantic staleness, exact-candidate evidence, and failed compare-and-swap visible before successful revalidation and publication.",
  "guard": "Textual merge success does not establish semantic correctness. Checks apply to the exact candidate they examined. Atomic publication must compare the captured target head; a moved head requires revalidation. Worktrees isolate files, not tool execution authority or external effects.",
  "slots": {
    "shared": {
      "x": 180,
      "y": 75,
      "width": 255,
      "font": 12
    },
    "interface": {
      "x": 102,
      "y": 160,
      "width": 130,
      "font": 11
    },
    "caller": {
      "x": 258,
      "y": 160,
      "width": 130,
      "font": 11
    },
    "base": {
      "x": 400,
      "y": 66,
      "width": 90,
      "font": 11
    },
    "head": {
      "x": 550,
      "y": 66,
      "width": 90,
      "font": 11
    },
    "worker": {
      "x": 530,
      "y": 195,
      "width": 110,
      "font": 11
    },
    "candidate": {
      "x": 660,
      "y": 195,
      "width": 110,
      "font": 11
    },
    "revalidate": {
      "x": 88,
      "y": 324,
      "width": 125,
      "font": 10
    },
    "build": {
      "x": 226,
      "y": 324,
      "width": 125,
      "font": 10
    },
    "checks": {
      "x": 364,
      "y": 324,
      "width": 125,
      "font": 10
    },
    "cas": {
      "x": 510,
      "y": 324,
      "width": 125,
      "font": 10
    },
    "published": {
      "x": 644,
      "y": 324,
      "width": 125,
      "font": 10
    },
    "loop": {
      "x": 299,
      "y": 268,
      "width": 270,
      "font": 11
    }
  },
  "defaults": {},
  "fields": [
    "Candidate identity",
    "Checks bound to",
    "Captured head",
    "Current target head",
    "Publication",
    "Isolation"
  ],
  "states": [
    {
      "label": "Isolate private edits from publication",
      "description": "Workers edit private worktrees from h0. Worktree isolation does not authorize publication or provide execution sandboxing; the original sandbox boundaries remain separate.",
      "text": {
        "shared": "shared target: h0",
        "interface": "interface revision h0",
        "caller": "private caller from h0",
        "base": "h_base = h0",
        "head": "h_head = h0",
        "worker": "private worker w1",
        "candidate": "candidate none",
        "revalidate": "not yet revalidated",
        "build": "no candidate",
        "checks": "not yet checked",
        "cas": "CAS expects h0",
        "published": "not published",
        "loop": "checks belong to an exact candidate"
      },
      "values": [
        "none",
        "No evidence",
        "h0",
        "h0",
        "No publication",
        "Private worktree; separate sandbox"
      ],
      "facts": {
        "candidate": "none",
        "checked_candidate": null,
        "expected_head": "h0",
        "target_head": "h0",
        "compared_head": null,
        "guard": false,
        "published": false,
        "head_moved": false,
        "textually_clean": false,
        "private_worktree_is_sandbox": false
      },
      "camera": [
        180,
        120
      ]
    },
    {
      "label": "Advance the shared interface",
      "description": "The shared target advances to h1 and changes an interface. When that advance completes, the private caller can be semantically stale even if its text still merges cleanly.",
      "text": {
        "shared": "shared target: h1",
        "interface": "interface revision h1",
        "caller": "private caller from h0",
        "base": "h_base = h0",
        "head": "h_head = h1",
        "worker": "private worker w1",
        "candidate": "candidate none",
        "revalidate": "not yet revalidated",
        "build": "no candidate",
        "checks": "not yet checked",
        "cas": "CAS expects h0",
        "published": "not published",
        "loop": "checks belong to an exact candidate"
      },
      "values": [
        "none",
        "No evidence",
        "h0",
        "h1",
        "No publication",
        "Private worktree; separate sandbox"
      ],
      "facts": {
        "candidate": "none",
        "checked_candidate": null,
        "expected_head": "h0",
        "target_head": "h1",
        "compared_head": null,
        "guard": false,
        "published": false,
        "head_moved": false,
        "textually_clean": false,
        "private_worktree_is_sandbox": false
      },
      "camera": [
        130,
        120
      ],
      "motion": {
        "label": "h1",
        "route": [
          [
            180,
            75
          ],
          [
            400,
            90
          ],
          [
            550,
            90
          ]
        ],
        "role": "memory",
        "lock_camera": false
      }
    },
    {
      "label": "Build a textually clean but stale candidate",
      "description": "Build illustrative candidate c1 without a textual conflict. When the build finishes, the stale dependency still prevents publication; clean text is not semantic validation.",
      "text": {
        "shared": "shared target: h1",
        "interface": "interface revision h1",
        "caller": "private caller from h0",
        "base": "h_base = h0",
        "head": "h_head = h1",
        "worker": "private worker w1",
        "candidate": "candidate c1",
        "revalidate": "dependency stale",
        "build": "c1 on h0",
        "checks": "not yet checked",
        "cas": "CAS expects h0",
        "published": "not published",
        "loop": "checks belong to an exact candidate"
      },
      "values": [
        "c1",
        "No evidence",
        "h0",
        "h1",
        "No publication",
        "Private worktree; separate sandbox"
      ],
      "facts": {
        "candidate": "c1",
        "checked_candidate": null,
        "expected_head": "h0",
        "target_head": "h1",
        "compared_head": null,
        "guard": false,
        "published": false,
        "head_moved": false,
        "textually_clean": true,
        "private_worktree_is_sandbox": false
      },
      "camera": [
        600,
        150
      ],
      "motion": {
        "label": "build c1",
        "route": [
          [
            530,
            160
          ],
          [
            660,
            160
          ]
        ],
        "role": "state",
        "lock_camera": false
      }
    },
    {
      "label": "Revalidate the changed dependency",
      "description": "Revalidate dependencies against captured h1 and repair the caller. When this action finishes, the build must use that revised state; old checks cannot validate a new candidate.",
      "text": {
        "shared": "shared target: h1",
        "interface": "interface revision h1",
        "caller": "private caller from h0",
        "base": "h_base = h0",
        "head": "h_head = h1",
        "worker": "private worker w1",
        "candidate": "candidate c1",
        "revalidate": "revalidated h1",
        "build": "c1 on h0",
        "checks": "not yet checked",
        "cas": "CAS expects h1",
        "published": "not published",
        "loop": "checks belong to an exact candidate"
      },
      "values": [
        "c1",
        "No evidence",
        "h1",
        "h1",
        "No publication",
        "Private worktree; separate sandbox"
      ],
      "facts": {
        "candidate": "c1",
        "checked_candidate": null,
        "expected_head": "h1",
        "target_head": "h1",
        "compared_head": null,
        "guard": false,
        "published": false,
        "head_moved": false,
        "textually_clean": true,
        "private_worktree_is_sandbox": false
      },
      "camera": [
        88,
        310
      ],
      "motion": {
        "label": "revalidate",
        "route": [
          [
            530,
            146
          ],
          [
            410,
            101
          ],
          [
            88,
            290
          ]
        ],
        "role": "state",
        "lock_camera": false
      }
    },
    {
      "label": "Build candidate c2 against captured h1",
      "description": "Build c2 against captured h1. When construction finishes, c2 has its own identity and still requires its own checks.",
      "text": {
        "shared": "shared target: h1",
        "interface": "interface revision h1",
        "caller": "private caller from h0",
        "base": "h_base = h0",
        "head": "h_head = h1",
        "worker": "private worker w1",
        "candidate": "candidate c2",
        "revalidate": "revalidated h1",
        "build": "c2 on h1",
        "checks": "not yet checked",
        "cas": "CAS expects h1",
        "published": "not published",
        "loop": "checks belong to an exact candidate"
      },
      "values": [
        "c2",
        "No evidence",
        "h1",
        "h1",
        "No publication",
        "Private worktree; separate sandbox"
      ],
      "facts": {
        "candidate": "c2",
        "checked_candidate": null,
        "expected_head": "h1",
        "target_head": "h1",
        "compared_head": null,
        "guard": false,
        "published": false,
        "head_moved": false,
        "textually_clean": true,
        "private_worktree_is_sandbox": false
      },
      "camera": [
        226,
        310
      ],
      "motion": {
        "label": "c2",
        "route": [
          [
            88,
            308
          ],
          [
            226,
            308
          ],
          [
            660,
            160
          ]
        ],
        "role": "state",
        "lock_camera": false
      }
    },
    {
      "label": "Check that exact candidate c2",
      "description": "Run checks on the exact c2 candidate. When they finish, evidence belongs to c2 at captured h1, not any later build.",
      "text": {
        "shared": "shared target: h1",
        "interface": "interface revision h1",
        "caller": "private caller from h0",
        "base": "h_base = h0",
        "head": "h_head = h1",
        "worker": "private worker w1",
        "candidate": "candidate c2",
        "revalidate": "revalidated h1",
        "build": "c2 on h1",
        "checks": "checks(c2) retained",
        "cas": "CAS expects h1",
        "published": "not published",
        "loop": "checks belong to an exact candidate"
      },
      "values": [
        "c2",
        "c2",
        "h1",
        "h1",
        "No publication",
        "Private worktree; separate sandbox"
      ],
      "facts": {
        "candidate": "c2",
        "checked_candidate": "c2",
        "expected_head": "h1",
        "target_head": "h1",
        "compared_head": null,
        "guard": false,
        "published": false,
        "head_moved": false,
        "textually_clean": true,
        "private_worktree_is_sandbox": false
      },
      "camera": [
        364,
        310
      ],
      "motion": {
        "label": "checks(c2)",
        "route": [
          [
            226,
            308
          ],
          [
            364,
            308
          ]
        ],
        "role": "state",
        "lock_camera": false
      }
    },
    {
      "label": "Move the target head during review",
      "description": "The shared target advances to h2 while c2 checks remain retained. When this advance finishes, the expected h1 no longer matches the target.",
      "text": {
        "shared": "shared target: h2",
        "interface": "interface revision h2",
        "caller": "private caller from h0",
        "base": "h_base = h0",
        "head": "h_head = h2",
        "worker": "private worker w1",
        "candidate": "candidate c2",
        "revalidate": "revalidated h1",
        "build": "c2 on h1",
        "checks": "checks(c2) retained",
        "cas": "CAS expects h1",
        "published": "not published",
        "loop": "head moved: revalidate"
      },
      "values": [
        "c2",
        "c2",
        "h1",
        "h2",
        "No publication",
        "Private worktree; separate sandbox"
      ],
      "facts": {
        "candidate": "c2",
        "checked_candidate": "c2",
        "expected_head": "h1",
        "target_head": "h2",
        "compared_head": null,
        "guard": false,
        "published": false,
        "head_moved": true,
        "textually_clean": true,
        "private_worktree_is_sandbox": false
      },
      "camera": [
        510,
        140
      ],
      "motion": {
        "label": "h2",
        "route": [
          [
            400,
            90
          ],
          [
            550,
            90
          ]
        ],
        "role": "memory",
        "lock_camera": false
      }
    },
    {
      "label": "Reject compare-and-swap at h1",
      "description": "Attempt atomic publication with expected head h1. When compare-and-swap observes h2, it rejects publication; checked c2 remains unpublished.",
      "text": {
        "shared": "shared target: h2",
        "interface": "interface revision h2",
        "caller": "private caller from h0",
        "base": "h_base = h0",
        "head": "h_head = h2",
        "worker": "private worker w1",
        "candidate": "candidate c2",
        "revalidate": "revalidated h1",
        "build": "c2 on h1",
        "checks": "checks(c2) retained",
        "cas": "CAS expects h1: reject",
        "published": "not published",
        "loop": "head moved: revalidate"
      },
      "values": [
        "c2",
        "c2",
        "h1",
        "h2",
        "No publication",
        "Private worktree; separate sandbox"
      ],
      "facts": {
        "candidate": "c2",
        "checked_candidate": "c2",
        "expected_head": "h1",
        "target_head": "h2",
        "compared_head": null,
        "guard": false,
        "published": false,
        "head_moved": true,
        "textually_clean": true,
        "private_worktree_is_sandbox": false
      },
      "camera": [
        535,
        180
      ],
      "motion": {
        "label": "CAS reject",
        "route": [
          [
            364,
            308
          ],
          [
            510,
            308
          ]
        ],
        "role": "fault",
        "lock_camera": false
      }
    },
    {
      "label": "Revalidate against the new target h2",
      "description": "Return to revalidation at captured h2. Earlier checks(c2) remain history; they cannot validate a new candidate against this changed target.",
      "text": {
        "shared": "shared target: h2",
        "interface": "interface revision h2",
        "caller": "private caller from h0",
        "base": "h_base = h0",
        "head": "h_head = h2",
        "worker": "private worker w1",
        "candidate": "candidate c2",
        "revalidate": "revalidated h2",
        "build": "c2 on h1",
        "checks": "checks(c2) retained",
        "cas": "CAS expects h2",
        "published": "not published",
        "loop": "head moved: revalidate"
      },
      "values": [
        "c2",
        "c2",
        "h2",
        "h2",
        "No publication",
        "Private worktree; separate sandbox"
      ],
      "facts": {
        "candidate": "c2",
        "checked_candidate": "c2",
        "expected_head": "h2",
        "target_head": "h2",
        "compared_head": null,
        "guard": false,
        "published": false,
        "head_moved": false,
        "textually_clean": true,
        "private_worktree_is_sandbox": false
      },
      "camera": [
        88,
        310
      ],
      "motion": {
        "label": "head h2",
        "route": [
          [
            510,
            290
          ],
          [
            510,
            274
          ],
          [
            88,
            274
          ],
          [
            88,
            289
          ]
        ],
        "role": "fault",
        "lock_camera": false
      }
    },
    {
      "label": "Build a fresh candidate c3",
      "description": "Build fresh c3 against captured h2. When construction finishes, c3 still has no checks of its own; retained c2 evidence cannot validate it.",
      "text": {
        "shared": "shared target: h2",
        "interface": "interface revision h2",
        "caller": "private caller from h0",
        "base": "h_base = h0",
        "head": "h_head = h2",
        "worker": "private worker w1",
        "candidate": "candidate c3",
        "revalidate": "revalidated h2",
        "build": "c3 on h2",
        "checks": "checks(c2) retained",
        "cas": "CAS expects h2",
        "published": "not published",
        "loop": "checks belong to an exact candidate"
      },
      "values": [
        "c3",
        "c2",
        "h2",
        "h2",
        "No publication",
        "Private worktree; separate sandbox"
      ],
      "facts": {
        "candidate": "c3",
        "checked_candidate": "c2",
        "expected_head": "h2",
        "target_head": "h2",
        "compared_head": null,
        "guard": false,
        "published": false,
        "head_moved": false,
        "textually_clean": true,
        "private_worktree_is_sandbox": false
      },
      "camera": [
        226,
        310
      ],
      "motion": {
        "label": "build c3",
        "route": [
          [
            88,
            308
          ],
          [
            226,
            308
          ]
        ],
        "role": "state",
        "lock_camera": false
      }
    },
    {
      "label": "Check that exact fresh candidate c3",
      "description": "Run checks on exact c3 against captured h2. Only when this action finishes do checks(c3) exist; c2 evidence is not transferred to c3.",
      "text": {
        "shared": "shared target: h2",
        "interface": "interface revision h2",
        "caller": "private caller from h0",
        "base": "h_base = h0",
        "head": "h_head = h2",
        "worker": "private worker w1",
        "candidate": "candidate c3",
        "revalidate": "revalidated h2",
        "build": "c3 on h2",
        "checks": "checks(c3) retained",
        "cas": "CAS expects h2",
        "published": "not published",
        "loop": "checks belong to an exact candidate"
      },
      "values": [
        "c3",
        "c3",
        "h2",
        "h2",
        "No publication",
        "Private worktree; separate sandbox"
      ],
      "facts": {
        "candidate": "c3",
        "checked_candidate": "c3",
        "expected_head": "h2",
        "target_head": "h2",
        "compared_head": null,
        "guard": false,
        "published": false,
        "head_moved": false,
        "textually_clean": true,
        "private_worktree_is_sandbox": false
      },
      "camera": [
        364,
        310
      ],
      "motion": {
        "label": "checks(c3)",
        "route": [
          [
            226,
            308
          ],
          [
            364,
            308
          ]
        ],
        "role": "state",
        "lock_camera": false
      }
    },
    {
      "label": "Publish only the checked candidate at the captured head",
      "description": "Compare-and-swap requires the target still equal captured h2 and checks bound to exact c3. Only when the guard succeeds may c3 become the shared published state.",
      "text": {
        "shared": "shared target: c3",
        "interface": "interface revision h2",
        "caller": "private caller from h0",
        "base": "h_base = h0",
        "head": "h_head = c3",
        "worker": "private worker w1",
        "candidate": "candidate c3",
        "revalidate": "revalidated h2",
        "build": "c3 on h2",
        "checks": "checks(c3) retained",
        "cas": "CAS h2 → c3: passed",
        "published": "published c3",
        "loop": "checks belong to an exact candidate"
      },
      "values": [
        "c3",
        "c3",
        "h2",
        "c3",
        "c3 published",
        "Private worktree; separate sandbox"
      ],
      "facts": {
        "candidate": "c3",
        "checked_candidate": "c3",
        "expected_head": "h2",
        "target_head": "c3",
        "compared_head": "h2",
        "guard": true,
        "published": true,
        "head_moved": false,
        "textually_clean": true,
        "private_worktree_is_sandbox": false
      },
      "camera": [
        610,
        180
      ],
      "motion": {
        "label": "publish c3",
        "route": [
          [
            364,
            308
          ],
          [
            510,
            308
          ],
          [
            644,
            308
          ]
        ],
        "role": "memory",
        "lock_camera": false
      }
    }
  ],
  "name": "original-v3-occ-worktrees"
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

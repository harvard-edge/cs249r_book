/* Original chapter artwork with inspectable, endpoint-driven systems state. */
(() => {
'use strict';
const spec = {
  "viewBox": "0 0 740 440",
  "asset": "fig-vol3-occ-worktrees.svg",
  "question": "Why can a checked private change still fail publication?",
  "intro": "Trace 12 guarded publication steps across private worktrees to see why stale dependencies and concurrent head updates require revalidation and fresh candidate checks.",
  "scope": "Three-panel optimistic concurrency control (OCC) diagram: Panel (a) shows isolated interface and caller worktrees inside separate execution sandboxes sharing repository storage; Panel (b) shows the commit DAG where base revision h0 branches into interface head h1 and caller worker w1; Panel (c) shows the 4-step guarded publication gate (1 revalidate, 2 build candidate, 3 checks, 4 atomic update) across target heads h0, h1, and h2 and candidate commits c1, c2, and c3.",
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
      "description": "At step 1 of 12 in Panel (a), the interface worker and caller worker check out linked Git worktrees from base revision h0 over shared repository storage. Each worktree isolates working files, index, and HEAD, while the dashed amber sandbox around each worktree enforces execution and write boundaries.",
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
        "none (editing in worktree)",
        "No evidence",
        "h0",
        "h0",
        "Unpublished (private worktrees at h0)",
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
      "description": "In Panel (b), the interface worker publishes commit h1 along the top arrow (interface change), advancing the shared target head from h0 to h1 while the caller worker w1 continues editing against the older interface contract at h_base = h0.",
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
        "h0 (caller base)",
        "h1 (interface changed)",
        "Unpublished (caller dependency stale vs h1)",
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
      "description": "In Panel (b), because caller worker w1 edited files disjoint from the interface commit h1, staging a merge into candidate c1 succeeds without textual conflict. However, Gate 1 (revalidate h_base vs. h_head) in Panel (c) detects the red dashed stale dependency on h0 and exits via request revision.",
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
        "c1 (textually clean, semantically stale)",
        "No evidence",
        "h0",
        "h1",
        "Blocked at Gate 1 (request revision)",
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
      "description": "In Panel (c) Gate 1 (1 revalidate), the integration service captures current target head h1, revalidates the caller's read dependencies against h1, and updates the caller patch to match the new interface signature.",
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
        "c1 (superseded)",
        "No evidence",
        "h1 (captured for attempt 2)",
        "h1",
        "Gate 1 passed at h1; ready to build c2",
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
      "description": "At Panel (c) Gate 2 (2 build candidate), the service stages a new immutable integration commit c2 combining captured head h1 with the revised caller changes. Candidate c2 has a new commit hash and no verification evidence yet.",
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
        "c2 (built on captured h1)",
        "No evidence (c2 unchecked)",
        "h1",
        "h1",
        "Gate 2 complete; awaiting Gate 3 checks",
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
      "description": "At Panel (c) Gate 3 (3 checks), the protected assessment sandbox runs acceptance tests directly on commit hash c2. The resulting verification receipt is cryptographically bound to exact candidate c2 at captured head h1.",
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
        "c2 (verified at captured h1)",
        "h1",
        "h1",
        "Gate 3 passed for c2; ready for Gate 4 CAS",
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
      "description": "While Gate 3 checks on c2 are finishing, a concurrent writer advances the shared target branch from h1 to h2 in Panel (b). Captured head h1 no longer matches the live target head h2.",
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
        "c2 (bound to old head h1)",
        "h1 (stale expected head)",
        "h2 (advanced concurrently)",
        "Target moved to h2 before CAS",
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
      "description": "At Panel (c) Gate 4 (4 atomic update), update-ref compares expected head h1 against current target head h2, sees h1 != h2, and aborts publication so c2 cannot overwrite h2.",
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
        "c2 (unpublished)",
        "c2",
        "h1",
        "h2",
        "Rejected at Gate 4 (CAS h1 != h2); loop to step 1",
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
      "description": "Following the dashed purple arrow (head moved: return to step 1) back to Gate 1, the service captures new target head h2 and revalidates the caller's dependencies against h2 rather than blindly re-running update-ref.",
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
        "c2 (superseded)",
        "c2 (historical only)",
        "h2 (captured for attempt 3)",
        "h2",
        "Gate 1 revalidated at h2",
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
      "description": "At Gate 2 (2 build candidate), the service stages fresh merge commit c3 combining captured head h2 with the worker branch. Because c3 is a new tree, prior checks bound to c2 cannot validate c3.",
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
        "c3 (built on captured h2)",
        "c2 (does not cover c3)",
        "h2",
        "h2",
        "Gate 2 built c3; requires fresh Gate 3 checks",
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
      "description": "At Gate 3 (3 checks), the assessment sandbox evaluates exact candidate c3, producing fresh verification evidence checks(c3) bound to commit c3 and captured head h2.",
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
        "c3 (verified at captured h2)",
        "h2",
        "h2",
        "Gate 3 passed for c3; ready for Gate 4 CAS",
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
      "description": "At Gate 4 (4 atomic update), update-ref verifies that the target head still equals expected h2 while the publication guard confirms active authority and checks(c3), atomically advancing the shared target to c3 (published with evidence).",
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
        "c3 (atomic ref update succeeded)",
        "c3 published with evidence",
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

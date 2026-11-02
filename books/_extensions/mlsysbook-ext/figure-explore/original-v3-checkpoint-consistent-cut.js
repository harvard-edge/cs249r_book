/* Original chapter artwork with inspectable, endpoint-driven systems state. */
(() => {
'use strict';
const spec = {
  "viewBox": "0 0 680 460",
  "asset": "checkpoint_consistent_cut.svg",
  "question": "When may a new checkpoint replace an earlier recovery cut?",
  "scope": "Illustrative boundaries r18 and r17 on the original compatible-cut and torn-pair panels. c0 is already published; c1 is the attempted new checkpoint. The case selector chooses alternatives, not successive states of one pair. Pending test_r18 and its reservation survive recovery.",
  "purpose": "Show referenced payload becoming durable before manifest publication, while an older recovery cut and unresolved obligations survive.",
  "guard": "A recovery cut must preserve compatible record and workspace payloads plus the retained suffix needed for replay. A torn pair is rejected. Replay folds recorded state; it neither repeats external effects nor settles an unknown tool outcome.",
  "slots": {
    "record": {
      "x": 115,
      "y": 106,
      "width": 175,
      "font": 12
    },
    "workspace": {
      "x": 335,
      "y": 106,
      "width": 175,
      "font": 12
    },
    "payload": {
      "x": 225,
      "y": 196.5,
      "width": 250,
      "font": 12
    },
    "manifest": {
      "x": 225,
      "y": 276.5,
      "width": 250,
      "font": 12
    },
    "suffix": {
      "x": 225,
      "y": 356.5,
      "width": 250,
      "font": 12
    },
    "pending": {
      "x": 225,
      "y": 426.5,
      "width": 390,
      "font": 12
    },
    "tornrecord": {
      "x": 560,
      "y": 85,
      "width": 180,
      "font": 12
    },
    "tornwork": {
      "x": 560,
      "y": 195,
      "width": 180,
      "font": 12
    },
    "tornstatus": {
      "x": 560,
      "y": 216,
      "width": 180,
      "font": 11
    },
    "reject": {
      "x": 560,
      "y": 296.5,
      "width": 180,
      "font": 12
    },
    "previous": {
      "x": 560,
      "y": 350,
      "width": 185,
      "font": 12
    },
    "fallback": {
      "x": 560,
      "y": 370,
      "width": 185,
      "font": 11
    }
  },
  "defaults": {},
  "fields": [
    "Earlier cut",
    "New manifest",
    "Record/workspace boundary",
    "Payload gate",
    "Recovered obligation",
    "Recovery action"
  ],
  "variants": [
    {
      "id": "compatible",
      "label": "A · Compatible new cut",
      "states": [
        {
          "label": "Keep the earlier published cut",
          "description": "The already published compatible cut c0 remains a recovery option. A new checkpoint under construction must not replace it.",
          "text": {
            "record": "new boundary not frozen",
            "workspace": "capture pending",
            "payload": "payload not yet durable",
            "manifest": "c1 not published",
            "suffix": "suffix not yet folded",
            "pending": "pending obligation remains durable",
            "tornrecord": "B: torn-pair reference",
            "tornwork": "old workspace reference",
            "tornstatus": "not the active case",
            "reject": "Torn pairs are ineligible",
            "previous": "c0 remains published",
            "fallback": "earlier compatible cut retained"
          },
          "values": [
            "c0 retained",
            "c1 unpublished",
            "capture pending",
            "Not admitted durable",
            "Pending test; reservation held",
            "Fold only; no external execution"
          ],
          "facts": {
            "c0_available": true,
            "c1_published": false,
            "payload_durable": false,
            "compatible": false,
            "restored": false,
            "pending_test": true,
            "reservation_held": true,
            "tool_reexecuted": false
          },
          "camera": [
            560,
            350
          ]
        },
        {
          "label": "Freeze the record boundary",
          "description": "Freeze a record snapshot at boundary r18, including references to workspace bytes. When capture finishes, the record exists but is not a published checkpoint.",
          "text": {
            "record": "r18 frozen",
            "workspace": "capture pending",
            "payload": "payload not yet durable",
            "manifest": "c1 not published",
            "suffix": "suffix not yet folded",
            "pending": "pending obligation remains durable",
            "tornrecord": "B: torn-pair reference",
            "tornwork": "old workspace reference",
            "tornstatus": "not the active case",
            "reject": "Torn pairs are ineligible",
            "previous": "c0 remains published",
            "fallback": "earlier compatible cut retained"
          },
          "values": [
            "c0 retained",
            "c1 unpublished",
            "capture pending",
            "Not admitted durable",
            "Pending test; reservation held",
            "Fold only; no external execution"
          ],
          "facts": {
            "c0_available": true,
            "c1_published": false,
            "payload_durable": false,
            "compatible": false,
            "restored": false,
            "pending_test": true,
            "reservation_held": true,
            "tool_reexecuted": false
          },
          "camera": [
            115,
            110
          ],
          "motion": {
            "label": "record r18",
            "route": [
              [
                115,
                90
              ],
              [
                115,
                145
              ],
              [
                225,
                145
              ],
              [
                225,
                192
              ]
            ],
            "role": "state",
            "lock_camera": false
          }
        },
        {
          "label": "Capture the matching workspace",
          "description": "Capture workspace bytes at the same r18 boundary. When capture finishes, the pair is compatible, but publication still waits for durable payload.",
          "text": {
            "record": "r18 frozen",
            "workspace": "r18 frozen",
            "payload": "payload not yet durable",
            "manifest": "c1 not published",
            "suffix": "suffix not yet folded",
            "pending": "pending obligation remains durable",
            "tornrecord": "B: torn-pair reference",
            "tornwork": "old workspace reference",
            "tornstatus": "not the active case",
            "reject": "Torn pairs are ineligible",
            "previous": "c0 remains published",
            "fallback": "earlier compatible cut retained"
          },
          "values": [
            "c0 retained",
            "c1 unpublished",
            "r18/r18",
            "Not admitted durable",
            "Pending test; reservation held",
            "Fold only; no external execution"
          ],
          "facts": {
            "c0_available": true,
            "c1_published": false,
            "payload_durable": false,
            "compatible": true,
            "restored": false,
            "pending_test": true,
            "reservation_held": true,
            "tool_reexecuted": false
          },
          "camera": [
            335,
            110
          ],
          "motion": {
            "label": "workspace r18",
            "route": [
              [
                335,
                90
              ],
              [
                335,
                145
              ],
              [
                225,
                145
              ],
              [
                225,
                192
              ]
            ],
            "role": "state",
            "lock_camera": false
          }
        },
        {
          "label": "Make the new payload durable",
          "description": "Persist all referenced payload bytes. Only when durability completes may a manifest safely name them.",
          "text": {
            "record": "r18 frozen",
            "workspace": "r18 frozen",
            "payload": "r18 payload durable",
            "manifest": "c1 not published",
            "suffix": "suffix not yet folded",
            "pending": "pending obligation remains durable",
            "tornrecord": "B: torn-pair reference",
            "tornwork": "old workspace reference",
            "tornstatus": "not the active case",
            "reject": "Torn pairs are ineligible",
            "previous": "c0 remains published",
            "fallback": "earlier compatible cut retained"
          },
          "values": [
            "c0 retained",
            "c1 unpublished",
            "r18/r18",
            "Durable",
            "Pending test; reservation held",
            "Fold only; no external execution"
          ],
          "facts": {
            "c0_available": true,
            "c1_published": false,
            "payload_durable": true,
            "compatible": true,
            "restored": false,
            "pending_test": true,
            "reservation_held": true,
            "tool_reexecuted": false
          },
          "camera": [
            225,
            290
          ],
          "motion": {
            "label": "durable",
            "route": [
              [
                225,
                192
              ],
              [
                225,
                218
              ]
            ],
            "role": "state",
            "lock_camera": false
          }
        },
        {
          "label": "Publish only a durable compatible cut",
          "description": "Append the c1 manifest only after the compatible payload is durable. When publication finishes, c1 becomes a selectable recovery cut.",
          "text": {
            "record": "r18 frozen",
            "workspace": "r18 frozen",
            "payload": "r18 payload durable",
            "manifest": "c1 published · r18/r18",
            "suffix": "suffix not yet folded",
            "pending": "pending obligation remains durable",
            "tornrecord": "B: torn-pair reference",
            "tornwork": "old workspace reference",
            "tornstatus": "not the active case",
            "reject": "Torn pairs are ineligible",
            "previous": "c0 remains published",
            "fallback": "earlier compatible cut retained"
          },
          "values": [
            "c0 retained",
            "c1 published",
            "r18/r18",
            "Durable",
            "Pending test; reservation held",
            "Fold only; no external execution"
          ],
          "facts": {
            "c0_available": true,
            "c1_published": true,
            "payload_durable": true,
            "compatible": true,
            "restored": false,
            "pending_test": true,
            "reservation_held": true,
            "tool_reexecuted": false
          },
          "camera": [
            225,
            290
          ],
          "motion": {
            "label": "manifest c1",
            "route": [
              [
                225,
                215
              ],
              [
                225,
                272
              ]
            ],
            "role": "state",
            "lock_camera": false
          }
        },
        {
          "label": "Select a cut after a crash",
          "description": "After a crash, select published c1 and its retained payload.",
          "text": {
            "record": "r18 frozen",
            "workspace": "r18 frozen",
            "payload": "r18 payload durable",
            "manifest": "c1 published · r18/r18",
            "suffix": "suffix not yet folded",
            "pending": "pending obligation remains durable",
            "tornrecord": "B: torn-pair reference",
            "tornwork": "old workspace reference",
            "tornstatus": "not the active case",
            "reject": "Torn pairs are ineligible",
            "previous": "c0 remains published",
            "fallback": "recover c1 + suffix"
          },
          "values": [
            "c0 retained",
            "c1 published",
            "r18/r18",
            "Durable",
            "Pending test; reservation held",
            "Fold only; no external execution"
          ],
          "facts": {
            "c0_available": true,
            "c1_published": true,
            "payload_durable": true,
            "compatible": true,
            "restored": false,
            "pending_test": true,
            "reservation_held": true,
            "tool_reexecuted": false
          },
          "camera": [
            225,
            290
          ],
          "motion": {
            "label": "c1",
            "route": [
              [
                225,
                272
              ],
              [
                225,
                330
              ]
            ],
            "role": "state",
            "lock_camera": false
          }
        },
        {
          "label": "Fold the recorded suffix",
          "description": "Restore the selected payload and fold recorded suffix events. When recovery completes, it has reconstructed recorded state without invoking a model or tool again.",
          "text": {
            "record": "r18 frozen",
            "workspace": "r18 frozen",
            "payload": "r18 payload durable",
            "manifest": "c1 published · r18/r18",
            "suffix": "recorded suffix folded",
            "pending": "pending obligation remains durable",
            "tornrecord": "B: torn-pair reference",
            "tornwork": "old workspace reference",
            "tornstatus": "not the active case",
            "reject": "Torn pairs are ineligible",
            "previous": "c0 remains published",
            "fallback": "recover c1 + suffix"
          },
          "values": [
            "c0 retained",
            "c1 published",
            "r18/r18",
            "Durable",
            "Pending test; reservation held",
            "Fold only; no external execution"
          ],
          "facts": {
            "c0_available": true,
            "c1_published": true,
            "payload_durable": true,
            "compatible": true,
            "restored": true,
            "pending_test": true,
            "reservation_held": true,
            "tool_reexecuted": false
          },
          "camera": [
            225,
            400
          ],
          "motion": {
            "label": "fold suffix",
            "route": [
              [
                225,
                350
              ],
              [
                225,
                375
              ],
              [
                225,
                420
              ]
            ],
            "role": "state",
            "lock_camera": false
          }
        },
        {
          "label": "Retain the unresolved test obligation",
          "description": "The recovered record still contains unresolved test_r18 and its held reservation. Recovery is not evidence that the test failed or that its budget can be released.",
          "text": {
            "record": "r18 frozen",
            "workspace": "r18 frozen",
            "payload": "r18 payload durable",
            "manifest": "c1 published · r18/r18",
            "suffix": "recorded suffix folded",
            "pending": "test_r18 pending · reservation held",
            "tornrecord": "B: torn-pair reference",
            "tornwork": "old workspace reference",
            "tornstatus": "not the active case",
            "reject": "Torn pairs are ineligible",
            "previous": "c0 remains published",
            "fallback": "recover c1 + suffix"
          },
          "values": [
            "c0 retained",
            "c1 published",
            "r18/r18",
            "Durable",
            "Pending test; reservation held",
            "Fold only; no external execution"
          ],
          "facts": {
            "c0_available": true,
            "c1_published": true,
            "payload_durable": true,
            "compatible": true,
            "restored": true,
            "pending_test": true,
            "reservation_held": true,
            "tool_reexecuted": false
          },
          "camera": [
            225,
            400
          ]
        }
      ]
    },
    {
      "id": "torn",
      "label": "B · Torn pair rejected",
      "states": [
        {
          "label": "Keep the earlier published cut",
          "description": "The already published compatible cut c0 remains a recovery option. A new checkpoint under construction must not replace it.",
          "text": {
            "record": "new boundary not frozen",
            "workspace": "A: compatible reference",
            "payload": "A: durability gate reference",
            "manifest": "c1 not published",
            "suffix": "suffix not yet folded",
            "pending": "pending obligation remains durable",
            "tornrecord": "B: torn-pair reference",
            "tornwork": "old workspace reference",
            "tornstatus": "not the active case",
            "reject": "Torn pairs are ineligible",
            "previous": "c0 remains published",
            "fallback": "earlier compatible cut retained"
          },
          "values": [
            "c0 retained",
            "c1 unpublished",
            "capture pending",
            "Not admitted durable",
            "Pending test; reservation held",
            "Fold only; no external execution"
          ],
          "facts": {
            "c0_available": true,
            "c1_published": false,
            "payload_durable": false,
            "compatible": false,
            "restored": false,
            "pending_test": true,
            "reservation_held": true,
            "tool_reexecuted": false
          },
          "camera": [
            560,
            350
          ]
        },
        {
          "label": "Freeze the record boundary",
          "description": "Freeze a record snapshot at boundary r18, including references to workspace bytes. When capture finishes, the record exists but is not a published checkpoint.",
          "text": {
            "record": "r18 frozen",
            "workspace": "A: compatible reference",
            "payload": "A: durability gate reference",
            "manifest": "c1 not published",
            "suffix": "suffix not yet folded",
            "pending": "pending obligation remains durable",
            "tornrecord": "B: record r18",
            "tornwork": "old workspace reference",
            "tornstatus": "not the active case",
            "reject": "Torn pairs are ineligible",
            "previous": "c0 remains published",
            "fallback": "earlier compatible cut retained"
          },
          "values": [
            "c0 retained",
            "c1 unpublished",
            "capture pending",
            "Not admitted durable",
            "Pending test; reservation held",
            "Fold only; no external execution"
          ],
          "facts": {
            "c0_available": true,
            "c1_published": false,
            "payload_durable": false,
            "compatible": false,
            "restored": false,
            "pending_test": true,
            "reservation_held": true,
            "tool_reexecuted": false
          },
          "camera": [
            560,
            90
          ],
          "motion": {
            "label": "record r18",
            "route": [
              [
                560,
                90
              ],
              [
                560,
                135
              ]
            ],
            "role": "state",
            "lock_camera": false
          }
        },
        {
          "label": "Detect old workspace bytes",
          "description": "The workspace is only r17 while the record references r18 bytes. When comparison finishes, this is a torn pair; timestamps or a new record alone cannot make it recoverable.",
          "text": {
            "record": "r18 frozen",
            "workspace": "A: compatible reference",
            "payload": "A: durability gate reference",
            "manifest": "c1 not published",
            "suffix": "suffix not yet folded",
            "pending": "pending obligation remains durable",
            "tornrecord": "B: record r18",
            "tornwork": "workspace r17",
            "tornstatus": "r18 bytes absent",
            "reject": "Torn pairs are ineligible",
            "previous": "c0 remains published",
            "fallback": "earlier compatible cut retained"
          },
          "values": [
            "c0 retained",
            "c1 unpublished",
            "r18/r17: reject",
            "Not admitted durable",
            "Pending test; reservation held",
            "Fold only; no external execution"
          ],
          "facts": {
            "c0_available": true,
            "c1_published": false,
            "payload_durable": false,
            "compatible": false,
            "restored": false,
            "pending_test": true,
            "reservation_held": true,
            "tool_reexecuted": false
          },
          "camera": [
            560,
            200
          ],
          "motion": {
            "label": "r17 ≠ r18",
            "route": [
              [
                560,
                195
              ],
              [
                560,
                250
              ],
              [
                560,
                292
              ]
            ],
            "role": "fault",
            "lock_camera": false
          }
        },
        {
          "label": "Reject the torn recovery pair",
          "description": "Reject the torn pair. When this check finishes, c1 remains unpublished and c0 remains available; recovery must not invent the missing r18 bytes.",
          "text": {
            "record": "r18 frozen",
            "workspace": "A: compatible reference",
            "payload": "A: durability gate reference",
            "manifest": "c1 not published",
            "suffix": "suffix not yet folded",
            "pending": "pending obligation remains durable",
            "tornrecord": "B: record r18",
            "tornwork": "workspace r17",
            "tornstatus": "r18 bytes absent",
            "reject": "Rejected: no c1 publication",
            "previous": "c0 remains published",
            "fallback": "earlier compatible cut retained"
          },
          "values": [
            "c0 retained",
            "c1 unpublished",
            "r18/r17: reject",
            "Not admitted durable",
            "Pending test; reservation held",
            "Fold only; no external execution"
          ],
          "facts": {
            "c0_available": true,
            "c1_published": false,
            "payload_durable": false,
            "compatible": false,
            "restored": false,
            "pending_test": true,
            "reservation_held": true,
            "tool_reexecuted": false
          },
          "camera": [
            560,
            310
          ],
          "motion": {
            "label": "reject",
            "route": [
              [
                560,
                216
              ],
              [
                560,
                292
              ]
            ],
            "role": "fault",
            "lock_camera": false
          }
        },
        {
          "label": "Publish only a durable compatible cut",
          "description": "The publication gate remains closed for this incompatible pair. No c1 manifest is admitted; the earlier c0 is still authoritative.",
          "text": {
            "record": "r18 frozen",
            "workspace": "A: compatible reference",
            "payload": "A: durability gate reference",
            "manifest": "c1 not published",
            "suffix": "suffix not yet folded",
            "pending": "pending obligation remains durable",
            "tornrecord": "B: record r18",
            "tornwork": "workspace r17",
            "tornstatus": "r18 bytes absent",
            "reject": "Rejected: no c1 publication",
            "previous": "c0 remains published",
            "fallback": "earlier compatible cut retained"
          },
          "values": [
            "c0 retained",
            "c1 unpublished",
            "r18/r17: reject",
            "Not admitted durable",
            "Pending test; reservation held",
            "Fold only; no external execution"
          ],
          "facts": {
            "c0_available": true,
            "c1_published": false,
            "payload_durable": false,
            "compatible": false,
            "restored": false,
            "pending_test": true,
            "reservation_held": true,
            "tool_reexecuted": false
          },
          "camera": [
            560,
            310
          ]
        },
        {
          "label": "Select a cut after a crash",
          "description": "After a crash, select the earlier published c0 and the available recorded suffix; do not select the rejected torn pair.",
          "text": {
            "record": "r18 frozen",
            "workspace": "A: compatible reference",
            "payload": "A: durability gate reference",
            "manifest": "c1 not published",
            "suffix": "suffix not yet folded",
            "pending": "pending obligation remains durable",
            "tornrecord": "B: record r18",
            "tornwork": "workspace r17",
            "tornstatus": "r18 bytes absent",
            "reject": "Rejected: no c1 publication",
            "previous": "c0 remains published",
            "fallback": "recover c0 + suffix"
          },
          "values": [
            "c0 retained",
            "c1 unpublished",
            "r18/r17: reject",
            "Not admitted durable",
            "Pending test; reservation held",
            "Fold only; no external execution"
          ],
          "facts": {
            "c0_available": true,
            "c1_published": false,
            "payload_durable": false,
            "compatible": false,
            "restored": false,
            "pending_test": true,
            "reservation_held": true,
            "tool_reexecuted": false
          },
          "camera": [
            560,
            310
          ],
          "motion": {
            "label": "c0",
            "route": [
              [
                560,
                350
              ],
              [
                440,
                350
              ],
              [
                440,
                330
              ],
              [
                225,
                350
              ]
            ],
            "role": "state",
            "lock_camera": false
          }
        },
        {
          "label": "Fold the recorded suffix",
          "description": "Restore the selected payload and fold recorded suffix events. When recovery completes, it has reconstructed recorded state without invoking a model or tool again.",
          "text": {
            "record": "r18 frozen",
            "workspace": "A: compatible reference",
            "payload": "A: durability gate reference",
            "manifest": "c1 not published",
            "suffix": "recorded suffix folded",
            "pending": "pending obligation remains durable",
            "tornrecord": "B: record r18",
            "tornwork": "workspace r17",
            "tornstatus": "r18 bytes absent",
            "reject": "Rejected: no c1 publication",
            "previous": "c0 remains published",
            "fallback": "recover c0 + suffix"
          },
          "values": [
            "c0 retained",
            "c1 unpublished",
            "r18/r17: reject",
            "Not admitted durable",
            "Pending test; reservation held",
            "Fold only; no external execution"
          ],
          "facts": {
            "c0_available": true,
            "c1_published": false,
            "payload_durable": false,
            "compatible": false,
            "restored": true,
            "pending_test": true,
            "reservation_held": true,
            "tool_reexecuted": false
          },
          "camera": [
            225,
            400
          ],
          "motion": {
            "label": "fold suffix",
            "route": [
              [
                225,
                350
              ],
              [
                225,
                375
              ],
              [
                225,
                420
              ]
            ],
            "role": "state",
            "lock_camera": false
          }
        },
        {
          "label": "Retain the unresolved test obligation",
          "description": "The recovered record still contains unresolved test_r18 and its held reservation. Recovery is not evidence that the test failed or that its budget can be released.",
          "text": {
            "record": "r18 frozen",
            "workspace": "A: compatible reference",
            "payload": "A: durability gate reference",
            "manifest": "c1 not published",
            "suffix": "recorded suffix folded",
            "pending": "test_r18 pending · reservation held",
            "tornrecord": "B: record r18",
            "tornwork": "workspace r17",
            "tornstatus": "r18 bytes absent",
            "reject": "Rejected: no c1 publication",
            "previous": "c0 remains published",
            "fallback": "recover c0 + suffix"
          },
          "values": [
            "c0 retained",
            "c1 unpublished",
            "r18/r17: reject",
            "Not admitted durable",
            "Pending test; reservation held",
            "Fold only; no external execution"
          ],
          "facts": {
            "c0_available": true,
            "c1_published": false,
            "payload_durable": false,
            "compatible": false,
            "restored": true,
            "pending_test": true,
            "reservation_held": true,
            "tool_reexecuted": false
          },
          "camera": [
            225,
            400
          ]
        }
      ]
    }
  ],
  "variantLabel": "Checkpoint case",
  "minWidth": 720,
  "name": "original-v3-checkpoint-consistent-cut"
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
 if(spec.variants){choice=document.createElement('label');choice.className='v3-original-choice';choice.textContent=spec.variantLabel+' ';const select=document.createElement('select');select.setAttribute('aria-label',spec.variantLabel);spec.variants.forEach(v=>{const o=document.createElement('option');o.value=v.id;o.textContent=v.label;select.append(o);});select.addEventListener('change',()=>{variant=select.value;frames().forEach((s,i)=>{Object.assign(steps[i],{label:s.label,description:s.description});const option=root.querySelector('.mlsys-original-sequence select').options[i];option.textContent=(i+1)+'. '+s.label;});player.go(0);});choice.append(select);}
 function follow(x,y){active=[x,y];const scale=svg.getBoundingClientRect().width/vb.width;if(canvas.scrollWidth>canvas.clientWidth)canvas.scrollLeft=Math.max(0,x*scale-canvas.clientWidth/2);if(canvas.scrollHeight>canvas.clientHeight)canvas.scrollTop=Math.max(0,y*scale-canvas.clientHeight/2);}
 function render(i,{stage,details}){
  if(!canvas.isConnected){if(choice)root.insertBefore(choice,root.querySelector('.mlsys-original-controls'));stage.append(cue,canvas,readout);}
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

/* Original chapter artwork with inspectable, endpoint-driven systems state. */
(() => {
'use strict';
const spec = {
  "viewBox": "0 0 680 460",
  "asset": "checkpoint_consistent_cut.svg",
  "question": "When may a new checkpoint replace an earlier recovery cut?",
  "intro": "Compare compatible and torn checkpoint cuts across 8 steps to see why payload bytes at boundary r18 must become durable before manifest c1 replaces published cut c0.",
  "scope": "Two recovery scenarios on the chapter's compatible-cut (Panel A) and torn-pair (Panel B) diagram: published checkpoint c0 is already available while the runtime attempts to capture new checkpoint c1 at boundary r18 (versus stale workspace boundary r17 in Panel B). Pending operation test_r18 and its budget reservation remain in doubt across recovery.",
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
          "description": "Before capturing a new checkpoint in Panel A across the 8 steps, the earlier published cut c0 and its recorded log suffix remain the active fallback boundary so a crash during capture never strands recovery.",
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
            "c0 retained (active fallback)",
            "c1 unpublished",
            "capture pending",
            "Not admitted durable",
            "Pending test_r18; reservation held",
            "Fallback to c0 + suffix if crash occurs"
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
          "description": "In Panel A's top-left purple box, freeze the control record snapshot at committed log boundary r18 (capturing budget reservations and in-flight operation test_r18). Because the workspace half is not yet captured, c1 remains unpublished.",
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
            "c0 retained (active fallback)",
            "c1 unpublished",
            "record r18 / workspace pending",
            "Not admitted durable",
            "Pending test_r18; reservation held",
            "Wait for matching workspace snapshot"
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
          "description": "In Panel A's top-right green box, capture the workspace tree ws_r18 at the exact same log boundary r18. Both halves now share boundary r18/r18 (a consistent cut), but the manifest still waits for payload durability.",
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
            "c0 retained (active fallback)",
            "c1 unpublished",
            "r18/r18 (consistent cut)",
            "Flushing bytes to storage",
            "Pending test_r18; reservation held",
            "Withhold manifest until bytes are durable"
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
          "description": "Flush both the r18 record snapshot and the r18 workspace bytes into durable storage within the declared failure domain (second row of Panel A). Data availability must precede manifest publication so a published checkpoint never points to missing bytes.",
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
            "c0 retained (active fallback)",
            "c1 ready to publish",
            "r18/r18 (consistent cut)",
            "Durable in failure domain",
            "Pending test_r18; reservation held",
            "Ready to append checkpoint manifest c1"
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
          "description": "Commit manifest c1 to the authoritative log (third row of Panel A), binding log boundary r18 to the verified hashes of the durable record and workspace snapshots. Checkpoint c1 is now eligible for crash recovery.",
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
            "c1 published in log",
            "r18/r18 (published)",
            "Durable and hash-verified",
            "Pending test_r18; reservation held",
            "c1 eligible as primary recovery base"
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
          "description": "After a worker crash, the recovery supervisor inspects the authoritative log, verifies manifest c1's cryptographic hashes and artifact availability, and selects published cut c1 plus the post-checkpoint event suffix.",
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
            "c1 selected for recovery",
            "r18/r18 (verified)",
            "Durable payload unpacked",
            "Pending test_r18; reservation held",
            "Unpack c1 base and fetch log suffix"
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
          "description": "Restore workspace ws_r18 and replay the recorded log suffix after r18 through the deterministic reducer (fourth row of Panel A). Reconstruction writes preserved byte streams without re-invoking the foundation model or external tools.",
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
            "c1 restored + suffix folded",
            "r18/r18 + suffix",
            "Durable",
            "Pending test_r18; reservation held",
            "Fold suffix only; 0 model/tool calls"
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
          "description": "At the bottom box of Panel A, folding the suffix reinstates pending operation test_r18 and its locked budget reservation. Recovery preserves in-doubt obligations for settlement rather than assuming the test failed or releasing its funds.",
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
            "c1 active",
            "r18/r18 + suffix",
            "Durable",
            "test_r18 in doubt; reservation locked",
            "Reconcile test_r18 before retry or release"
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
          "description": "Before evaluating the candidate pair in Panel B across the 8 steps, earlier compatible checkpoint c0 and its log suffix remain durably published as the fallback recovery cut.",
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
            "c0 retained (active fallback)",
            "c1 unpublished",
            "capture pending",
            "Not admitted durable",
            "Pending test_r18; reservation held",
            "Fallback to c0 + suffix if capture fails"
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
          "description": "In Panel B's top purple box, the runtime captures a newer control record at boundary r18 that references workspace files modified by patch r18.",
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
            "c0 retained (active fallback)",
            "c1 unpublished",
            "record r18 / workspace unverified",
            "Not admitted durable",
            "Pending test_r18; reservation held",
            "Check workspace snapshot boundary"
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
          "description": "In Panel B's middle green box, the available workspace snapshot is still at older revision r17 and lacks the r18 payload bytes referenced by the record, forming a torn state that never existed at one boundary.",
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
            "c0 retained (active fallback)",
            "c1 unpublished",
            "r18/r17 mismatch (torn pair)",
            "Missing r18 workspace bytes",
            "Pending test_r18; reservation held",
            "Flag boundary mismatch r18 != r17"
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
          "description": "In Panel B's red box, the runtime rejects the r18/r17 pair because restoring a forward control record over an obsolete directory tree would corrupt state.",
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
            "c0 retained (active fallback)",
            "c1 rejected",
            "r18/r17: rejected",
            "Not admitted durable",
            "Pending test_r18; reservation held",
            "Refuse torn restoration; keep c0"
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
          "description": "Because the r18/r17 capture failed the compatibility and payload-availability gate, manifest c1 is never appended to the log; published checkpoint c0 remains the latest authoritative cut.",
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
            "c0 authoritative in log",
            "c1 unpublished (blocked)",
            "r18/r17: rejected",
            "Gate closed (missing r18 bytes)",
            "Pending test_r18; reservation held",
            "Retain c0 as latest valid checkpoint"
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
          "description": "Following a crash, the recovery supervisor ignores the uncommitted r18 staging files and selects the earlier published cut c0 at the bottom of Panel B together with its durable log suffix.",
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
            "c0 selected for recovery",
            "c1 ignored (unpublished)",
            "c0 compatible base selected",
            "c0 payload durable",
            "Pending test_r18; reservation held",
            "Fallback to published c0 + recorded suffix"
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
          "description": "Unpack compatible cut c0 and deterministically fold the preserved suffix events and workspace deltas forward without invoking the model or re-running external tools.",
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
            "c0 unpacked + suffix folded",
            "c1 unpublished",
            "c0 + recorded suffix",
            "c0 + suffix bytes verified",
            "Pending test_r18; reservation held",
            "Fold suffix from c0; 0 model/tool calls"
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
          "description": "Folding the suffix from c0 reconstructs the exact in-doubt test_r18 entry and its locked budget reservation at the bottom of Panel A, keeping the obligation owned until endpoint evidence settles it.",
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
            "c0 recovery complete",
            "c1 unpublished",
            "c0 + recorded suffix",
            "c0 + suffix bytes verified",
            "test_r18 in doubt; reservation locked",
            "Reconcile test_r18 before retry or release"
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
 if(spec.variants){choice=document.createElement('label');choice.className='v3-original-choice';choice.textContent=spec.variantLabel+' ';const select=document.createElement('select');select.setAttribute('aria-label',spec.variantLabel);spec.variants.forEach(v=>{const o=document.createElement('option');o.value=v.id;o.textContent=v.label;select.append(o);});select.addEventListener('change',()=>{variant=select.value;frames().forEach((s,i)=>{Object.assign(steps[i],{label:s.label,description:s.description});const option=root.querySelector('.mlsys-original-sequence select').options[i];option.textContent=(i+1)+' / '+steps.length+' · '+s.label;});player.go(0);});choice.append(select);}
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

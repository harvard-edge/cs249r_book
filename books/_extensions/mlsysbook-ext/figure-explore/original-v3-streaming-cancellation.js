/* Original chapter artwork with inspectable, endpoint-driven systems state. */
(() => {
'use strict';
const spec = {
  "viewBox": "0 0 720 362",
  "asset": "streaming_token_cancellation.svg",
  "question": "What does local rejection stop—and what needs provider cooperation?",
  "scope": "Two separate outcomes on the original stream/parser/cancellation diagram. The source example rejects token 96 under a 4,096-token allowance. Up to 4,000 later steps can be avoided only when cancellation is honored. Partial proposals never reach a tool.",
  "purpose": "Separate immediate local refusal from conditional provider stop and later KV release.",
  "guard": "A cancel request is not proof of cancellation. Remote KV may be reclaimed only when its consumers have stopped. No latency or realized saving is inferred from token allowance. The two cases are alternatives, not successive outcomes for one request.",
  "slots": {
    "service": {
      "x": 20,
      "y": 58,
      "width": 125,
      "font": 12
    },
    "budget": {
      "x": 20,
      "y": 74,
      "width": 125,
      "font": 10
    },
    "skipped": {
      "x": 426,
      "y": 67,
      "width": 235,
      "font": 12
    },
    "parser0": {
      "x": 185,
      "y": 218,
      "width": 80,
      "font": 11
    },
    "parser1": {
      "x": 260,
      "y": 218,
      "width": 80,
      "font": 11
    },
    "parser2": {
      "x": 410,
      "y": 218,
      "width": 100,
      "font": 11
    },
    "p0": {
      "x": 185,
      "y": 236,
      "width": 75,
      "font": 10
    },
    "p1": {
      "x": 260,
      "y": 236,
      "width": 75,
      "font": 10
    },
    "p2": {
      "x": 410,
      "y": 236,
      "width": 110,
      "font": 10
    },
    "cancel": {
      "x": 600,
      "y": 152,
      "width": 120,
      "font": 12
    },
    "cancel2": {
      "x": 600,
      "y": 168,
      "width": 120,
      "font": 11
    },
    "local1": {
      "x": 150,
      "y": 298,
      "width": 260,
      "font": 12
    },
    "local2": {
      "x": 150,
      "y": 318,
      "width": 260,
      "font": 12
    },
    "local3": {
      "x": 150,
      "y": 338,
      "width": 260,
      "font": 12
    },
    "remote1": {
      "x": 500,
      "y": 298,
      "width": 205,
      "font": 11
    },
    "remote2": {
      "x": 500,
      "y": 318,
      "width": 205,
      "font": 11
    },
    "remote3": {
      "x": 500,
      "y": 338,
      "width": 205,
      "font": 11
    }
  },
  "defaults": {},
  "fields": [
    "Local proposal",
    "Cancellation request",
    "Provider confirmation",
    "Remote decoding",
    "KV ownership"
  ],
  "variants": [
    {
      "id": "honored",
      "label": "Cancellation honored",
      "states": [
        {
          "label": "Receive a quarantined stream",
          "description": "The runtime opens a quarantined stream buffer under a 4,096-token output allowance; parser state starts at q0 while provider decoding and KV allocation remain active.",
          "text": {
            "service": "Decode active",
            "budget": "allowance: 4,096 tokens",
            "skipped": "Conditional saving: not established",
            "parser0": "q0: waiting",
            "parser1": "waiting",
            "parser2": "waiting",
            "p0": "not received",
            "p1": "not received",
            "p2": "not received",
            "cancel": "not sent",
            "cancel2": "no request",
            "local1": "partial proposal never dispatched",
            "local2": "validation pending",
            "local3": "no tool admission",
            "remote1": "decode still active",
            "remote2": "batch departure not established",
            "remote3": "KV held until consumers stop"
          },
          "values": [
            "Quarantined; never dispatched",
            "Not sent",
            "Unconfirmed",
            "May remain active",
            "Held"
          ],
          "facts": {
            "rejected": false,
            "dispatched": false,
            "cancel_sent": false,
            "honored": false,
            "remote_stopped": false,
            "kv_released": false,
            "kv_held": true
          },
          "camera": [
            185,
            210
          ]
        },
        {
          "label": "Hold the opening brace",
          "description": "Token 1 ('{') advances the local stream validator from q0 to q1; when validation finishes, the fragment remains held in quarantine because parsing is not authorization.",
          "text": {
            "service": "Decode active",
            "budget": "allowance: 4,096 tokens",
            "skipped": "Conditional saving: not established",
            "parser0": "q0 → q1",
            "parser1": "waiting",
            "parser2": "waiting",
            "p0": "held",
            "p1": "not received",
            "p2": "not received",
            "cancel": "not sent",
            "cancel2": "no request",
            "local1": "partial proposal never dispatched",
            "local2": "validation pending",
            "local3": "no tool admission",
            "remote1": "decode still active",
            "remote2": "batch departure not established",
            "remote3": "KV held until consumers stop"
          },
          "values": [
            "Quarantined; never dispatched",
            "Not sent",
            "Unconfirmed",
            "May remain active",
            "Held"
          ],
          "facts": {
            "rejected": false,
            "dispatched": false,
            "cancel_sent": false,
            "honored": false,
            "remote_stopped": false,
            "kv_released": false,
            "kv_held": true
          },
          "camera": [
            185,
            210
          ],
          "motion": {
            "label": "{",
            "route": [
              [
                185,
                143
              ],
              [
                185,
                218
              ]
            ],
            "role": "memory",
            "lock_camera": false
          }
        },
        {
          "label": "Hold a permitted key",
          "description": "Token 2 ('\"query\"') is permitted by the schema and advances the parser from q1 to q2; tokens 3 through 95 stay quarantined until a complete object is validated.",
          "text": {
            "service": "Decode active",
            "budget": "allowance: 4,096 tokens",
            "skipped": "Conditional saving: not established",
            "parser0": "q0 → q1",
            "parser1": "q1 → q2",
            "parser2": "waiting",
            "p0": "held",
            "p1": "held",
            "p2": "not received",
            "cancel": "not sent",
            "cancel2": "no request",
            "local1": "partial proposal never dispatched",
            "local2": "validation pending",
            "local3": "no tool admission",
            "remote1": "decode still active",
            "remote2": "batch departure not established",
            "remote3": "KV held until consumers stop"
          },
          "values": [
            "Quarantined; never dispatched",
            "Not sent",
            "Unconfirmed",
            "May remain active",
            "Held"
          ],
          "facts": {
            "rejected": false,
            "dispatched": false,
            "cancel_sent": false,
            "honored": false,
            "remote_stopped": false,
            "kv_released": false,
            "kv_held": true
          },
          "camera": [
            260,
            210
          ],
          "motion": {
            "label": "query",
            "route": [
              [
                260,
                143
              ],
              [
                260,
                218
              ]
            ],
            "role": "memory",
            "lock_camera": false
          }
        },
        {
          "label": "Reject the forbidden key locally",
          "description": "At token 96, key '\"force\":' violates the allowed schema; the validator enters REJECTED, records a local status failure, and refuses tool dispatch regardless of provider cooperation.",
          "text": {
            "service": "Decode active",
            "budget": "allowance: 4,096 tokens",
            "skipped": "Conditional saving: not established",
            "parser0": "q0 → q1",
            "parser1": "q1 → q2",
            "parser2": "REJECTED",
            "p0": "held",
            "p1": "held",
            "p2": "key not in schema",
            "cancel": "not sent",
            "cancel2": "no request",
            "local1": "partial proposal never dispatched",
            "local2": "rejection recorded",
            "local3": "error offered next turn",
            "remote1": "decode still active",
            "remote2": "batch departure not established",
            "remote3": "KV held until consumers stop"
          },
          "values": [
            "Rejected; never dispatched",
            "Not sent",
            "Unconfirmed",
            "May remain active",
            "Held"
          ],
          "facts": {
            "rejected": true,
            "dispatched": false,
            "cancel_sent": false,
            "honored": false,
            "remote_stopped": false,
            "kv_released": false,
            "kv_held": true
          },
          "camera": [
            410,
            210
          ],
          "motion": {
            "label": "force",
            "route": [
              [
                410,
                143
              ],
              [
                410,
                218
              ]
            ],
            "role": "fault",
            "lock_camera": false
          }
        },
        {
          "label": "Send cancellation upstream",
          "description": "The runtime sends a cancel request along the upstream control path to stop the remaining 4,096 - 96 = 4,000 decode steps; sending alone does not prove remote stop.",
          "text": {
            "service": "Decode active",
            "budget": "allowance: 4,096 tokens",
            "skipped": "Conditional saving: not established",
            "parser0": "q0 → q1",
            "parser1": "q1 → q2",
            "parser2": "REJECTED",
            "p0": "held",
            "p1": "held",
            "p2": "key not in schema",
            "cancel": "cancel sent",
            "cancel2": "request in flight",
            "local1": "partial proposal never dispatched",
            "local2": "rejection recorded",
            "local3": "error offered next turn",
            "remote1": "decode still active",
            "remote2": "batch departure not established",
            "remote3": "KV held until consumers stop"
          },
          "values": [
            "Rejected; never dispatched",
            "Sent",
            "Unconfirmed",
            "May remain active",
            "Held"
          ],
          "facts": {
            "rejected": true,
            "dispatched": false,
            "cancel_sent": true,
            "honored": false,
            "remote_stopped": false,
            "kv_released": false,
            "kv_held": true
          },
          "camera": [
            565,
            220
          ],
          "marks": [
            {
              "x": 350,
              "y": 158,
              "r": 18,
              "color": "#d97706"
            }
          ],
          "motion": {
            "label": "cancel",
            "route": [
              [
                470,
                222
              ],
              [
                590,
                222
              ],
              [
                590,
                82
              ]
            ],
            "role": "fault",
            "lock_camera": false
          }
        },
        {
          "label": "Wait for provider confirmation",
          "description": "The runtime awaits provider confirmation while the local proposal stays rejected; until the service confirms stop, remote decoding and KV ownership remain active.",
          "text": {
            "service": "Decode active",
            "budget": "allowance: 4,096 tokens",
            "skipped": "Conditional saving: not established",
            "parser0": "q0 → q1",
            "parser1": "q1 → q2",
            "parser2": "REJECTED",
            "p0": "held",
            "p1": "held",
            "p2": "key not in schema",
            "cancel": "cancel sent",
            "cancel2": "unconfirmed",
            "local1": "partial proposal never dispatched",
            "local2": "rejection recorded",
            "local3": "error offered next turn",
            "remote1": "decode still active",
            "remote2": "batch departure not established",
            "remote3": "KV held until consumers stop"
          },
          "values": [
            "Rejected; never dispatched",
            "Sent",
            "Unconfirmed",
            "May remain active",
            "Held"
          ],
          "facts": {
            "rejected": true,
            "dispatched": false,
            "cancel_sent": true,
            "honored": false,
            "remote_stopped": false,
            "kv_released": false,
            "kv_held": true
          },
          "camera": [
            565,
            220
          ],
          "marks": [
            {
              "x": 560,
              "y": 158,
              "r": 22,
              "color": "#d97706"
            }
          ]
        },
        {
          "label": "Observe the provider outcome",
          "description": "When the provider honors cancellation at step 96, remote decoding stops, avoiding up to 4,000 decode steps, and the request leaves the continuous batch.",
          "text": {
            "service": "Decode stopped",
            "budget": "allowance: 4,096 tokens",
            "skipped": "Up to 4,000 decode steps not run",
            "parser0": "q0 → q1",
            "parser1": "q1 → q2",
            "parser2": "REJECTED",
            "p0": "held",
            "p1": "held",
            "p2": "key not in schema",
            "cancel": "cancel sent",
            "cancel2": "honored",
            "local1": "partial proposal never dispatched",
            "local2": "rejection recorded",
            "local3": "error offered next turn",
            "remote1": "decode stops at step 96",
            "remote2": "request leaves batch",
            "remote3": "KV held until consumers stop"
          },
          "values": [
            "Rejected; never dispatched",
            "Sent",
            "Honored",
            "Stopped",
            "Held"
          ],
          "facts": {
            "rejected": true,
            "dispatched": false,
            "cancel_sent": true,
            "honored": true,
            "remote_stopped": true,
            "kv_released": false,
            "kv_held": true
          },
          "camera": [
            600,
            300
          ],
          "marks": [
            {
              "x": 560,
              "y": 300,
              "r": 22,
              "color": "#0284c7"
            }
          ],
          "motion": {
            "label": "stopped",
            "route": [
              [
                590,
                82
              ],
              [
                650,
                82
              ],
              [
                650,
                298
              ],
              [
                500,
                298
              ]
            ],
            "role": "memory",
            "lock_camera": false
          }
        },
        {
          "label": "Resolve remote resource ownership",
          "description": "Only after decoding and all device consumers stop at step 96 may the provider release KV cache memory; when release finishes, the request relinquishes those blocks.",
          "text": {
            "service": "Decode stopped",
            "budget": "allowance: 4,096 tokens",
            "skipped": "Up to 4,000 decode steps not run",
            "parser0": "q0 → q1",
            "parser1": "q1 → q2",
            "parser2": "REJECTED",
            "p0": "held",
            "p1": "held",
            "p2": "key not in schema",
            "cancel": "cancel sent",
            "cancel2": "honored",
            "local1": "partial proposal never dispatched",
            "local2": "rejection recorded",
            "local3": "error offered next turn",
            "remote1": "decode stops at step 96",
            "remote2": "request leaves batch",
            "remote3": "KV blocks released"
          },
          "values": [
            "Rejected; never dispatched",
            "Sent",
            "Honored",
            "Stopped",
            "Released after stop"
          ],
          "facts": {
            "rejected": true,
            "dispatched": false,
            "cancel_sent": true,
            "honored": true,
            "remote_stopped": true,
            "kv_released": true,
            "kv_held": false
          },
          "camera": [
            600,
            300
          ],
          "marks": [
            {
              "x": 560,
              "y": 336,
              "r": 22,
              "color": "#16a34a"
            }
          ],
          "motion": {
            "label": "release",
            "route": [
              [
                500,
                298
              ],
              [
                500,
                338
              ]
            ],
            "role": "memory",
            "lock_camera": false
          }
        }
      ]
    },
    {
      "id": "not-honored",
      "label": "Cancellation not honored",
      "states": [
        {
          "label": "Receive a quarantined stream",
          "description": "The runtime quarantines partial JSON under a 4,096-token output allowance while all three parser nodes wait (q0: waiting). Provider decoding and KV allocation remain active, and no fragment is eligible for tool dispatch.",
          "text": {
            "service": "Decode active",
            "budget": "allowance: 4,096 tokens",
            "skipped": "Conditional saving: not established",
            "parser0": "q0: waiting",
            "parser1": "waiting",
            "parser2": "waiting",
            "p0": "not received",
            "p1": "not received",
            "p2": "not received",
            "cancel": "not sent",
            "cancel2": "no request",
            "local1": "partial proposal never dispatched",
            "local2": "validation pending",
            "local3": "no tool admission",
            "remote1": "decode still active",
            "remote2": "batch departure not established",
            "remote3": "KV held until consumers stop"
          },
          "values": [
            "Quarantined; never dispatched",
            "Not sent",
            "Unconfirmed",
            "May remain active",
            "Held"
          ],
          "facts": {
            "rejected": false,
            "dispatched": false,
            "cancel_sent": false,
            "honored": false,
            "remote_stopped": false,
            "kv_released": false,
            "kv_held": true
          },
          "camera": [
            185,
            210
          ]
        },
        {
          "label": "Hold the opening brace",
          "description": "Streaming token { arrives at the first incremental parser box and advances state q0 → q1 (held). Local parsing is not execution authorization, so the partial proposal stays quarantined.",
          "text": {
            "service": "Decode active",
            "budget": "allowance: 4,096 tokens",
            "skipped": "Conditional saving: not established",
            "parser0": "q0 → q1",
            "parser1": "waiting",
            "parser2": "waiting",
            "p0": "held",
            "p1": "not received",
            "p2": "not received",
            "cancel": "not sent",
            "cancel2": "no request",
            "local1": "partial proposal never dispatched",
            "local2": "validation pending",
            "local3": "no tool admission",
            "remote1": "decode still active",
            "remote2": "batch departure not established",
            "remote3": "KV held until consumers stop"
          },
          "values": [
            "Quarantined; never dispatched",
            "Not sent",
            "Unconfirmed",
            "May remain active",
            "Held"
          ],
          "facts": {
            "rejected": false,
            "dispatched": false,
            "cancel_sent": false,
            "honored": false,
            "remote_stopped": false,
            "kv_released": false,
            "kv_held": true
          },
          "camera": [
            185,
            210
          ],
          "motion": {
            "label": "{",
            "route": [
              [
                185,
                143
              ],
              [
                185,
                218
              ]
            ],
            "role": "memory",
            "lock_camera": false
          }
        },
        {
          "label": "Hold a permitted key",
          "description": "The permitted schema key \"query\" arrives at the second parser box and advances state q1 → q2 (held). Because the JSON object is still incomplete and unvalidated, the runtime keeps the proposal quarantined.",
          "text": {
            "service": "Decode active",
            "budget": "allowance: 4,096 tokens",
            "skipped": "Conditional saving: not established",
            "parser0": "q0 → q1",
            "parser1": "q1 → q2",
            "parser2": "waiting",
            "p0": "held",
            "p1": "held",
            "p2": "not received",
            "cancel": "not sent",
            "cancel2": "no request",
            "local1": "partial proposal never dispatched",
            "local2": "validation pending",
            "local3": "no tool admission",
            "remote1": "decode still active",
            "remote2": "batch departure not established",
            "remote3": "KV held until consumers stop"
          },
          "values": [
            "Quarantined; never dispatched",
            "Not sent",
            "Unconfirmed",
            "May remain active",
            "Held"
          ],
          "facts": {
            "rejected": false,
            "dispatched": false,
            "cancel_sent": false,
            "honored": false,
            "remote_stopped": false,
            "kv_released": false,
            "kv_held": true
          },
          "camera": [
            260,
            210
          ],
          "motion": {
            "label": "query",
            "route": [
              [
                260,
                143
              ],
              [
                260,
                218
              ]
            ],
            "role": "memory",
            "lock_camera": false
          }
        },
        {
          "label": "Reject the forbidden key locally",
          "description": "At decode step 96, forbidden key \"force\" hits the third parser box and triggers REJECTED (key not in schema). The local runtime records the rejection and blocks tool dispatch immediately, regardless of provider cooperation.",
          "text": {
            "service": "Decode active",
            "budget": "allowance: 4,096 tokens",
            "skipped": "Conditional saving: not established",
            "parser0": "q0 → q1",
            "parser1": "q1 → q2",
            "parser2": "REJECTED",
            "p0": "held",
            "p1": "held",
            "p2": "key not in schema",
            "cancel": "not sent",
            "cancel2": "no request",
            "local1": "partial proposal never dispatched",
            "local2": "rejection recorded",
            "local3": "error offered next turn",
            "remote1": "decode still active",
            "remote2": "batch departure not established",
            "remote3": "KV held until consumers stop"
          },
          "values": [
            "Rejected; never dispatched",
            "Not sent",
            "Unconfirmed",
            "May remain active",
            "Held"
          ],
          "facts": {
            "rejected": true,
            "dispatched": false,
            "cancel_sent": false,
            "honored": false,
            "remote_stopped": false,
            "kv_released": false,
            "kv_held": true
          },
          "camera": [
            410,
            210
          ],
          "motion": {
            "label": "force",
            "route": [
              [
                410,
                143
              ],
              [
                410,
                218
              ]
            ],
            "role": "fault",
            "lock_camera": false
          }
        },
        {
          "label": "Send cancellation upstream",
          "description": "The runtime sends a cancel signal along the dashed upstream control path to the model service. Sending the request marks cancel as sent (unconfirmed) without yet proving that remote decoding stopped.",
          "text": {
            "service": "Decode active",
            "budget": "allowance: 4,096 tokens",
            "skipped": "Conditional saving: not established",
            "parser0": "q0 → q1",
            "parser1": "q1 → q2",
            "parser2": "REJECTED",
            "p0": "held",
            "p1": "held",
            "p2": "key not in schema",
            "cancel": "cancel sent",
            "cancel2": "request in flight",
            "local1": "partial proposal never dispatched",
            "local2": "rejection recorded",
            "local3": "error offered next turn",
            "remote1": "decode still active",
            "remote2": "batch departure not established",
            "remote3": "KV held until consumers stop"
          },
          "values": [
            "Rejected; never dispatched",
            "Sent",
            "Unconfirmed",
            "May remain active",
            "Held"
          ],
          "facts": {
            "rejected": true,
            "dispatched": false,
            "cancel_sent": true,
            "honored": false,
            "remote_stopped": false,
            "kv_released": false,
            "kv_held": true
          },
          "camera": [
            565,
            220
          ],
          "marks": [
            {
              "x": 350,
              "y": 158,
              "r": 18,
              "color": "#d97706"
            }
          ],
          "motion": {
            "label": "cancel",
            "route": [
              [
                470,
                222
              ],
              [
                590,
                222
              ],
              [
                590,
                82
              ]
            ],
            "role": "fault",
            "lock_camera": false
          }
        },
        {
          "label": "Wait for provider confirmation",
          "description": "Before provider confirmation arrives, the local proposal stays rejected and never dispatched, while remote decode status remains active and KV ownership remains held.",
          "text": {
            "service": "Decode active",
            "budget": "allowance: 4,096 tokens",
            "skipped": "Conditional saving: not established",
            "parser0": "q0 → q1",
            "parser1": "q1 → q2",
            "parser2": "REJECTED",
            "p0": "held",
            "p1": "held",
            "p2": "key not in schema",
            "cancel": "cancel sent",
            "cancel2": "unconfirmed",
            "local1": "partial proposal never dispatched",
            "local2": "rejection recorded",
            "local3": "error offered next turn",
            "remote1": "decode still active",
            "remote2": "batch departure not established",
            "remote3": "KV held until consumers stop"
          },
          "values": [
            "Rejected; never dispatched",
            "Sent",
            "Unconfirmed",
            "May remain active",
            "Held"
          ],
          "facts": {
            "rejected": true,
            "dispatched": false,
            "cancel_sent": true,
            "honored": false,
            "remote_stopped": false,
            "kv_released": false,
            "kv_held": true
          },
          "camera": [
            565,
            220
          ],
          "marks": [
            {
              "x": 560,
              "y": 158,
              "r": 22,
              "color": "#d97706"
            }
          ]
        },
        {
          "label": "Observe the provider outcome",
          "description": "In this provider case, cancellation at step 96 is not honored: remote decoding may continue toward the 4,096-token allowance, so the 4,000 skipped decode steps are not established.",
          "text": {
            "service": "Decode active",
            "budget": "allowance: 4,096 tokens",
            "skipped": "Conditional saving: not established",
            "parser0": "q0 → q1",
            "parser1": "q1 → q2",
            "parser2": "REJECTED",
            "p0": "held",
            "p1": "held",
            "p2": "key not in schema",
            "cancel": "cancel sent",
            "cancel2": "not honored",
            "local1": "partial proposal never dispatched",
            "local2": "rejection recorded",
            "local3": "error offered next turn",
            "remote1": "decode may continue",
            "remote2": "batch departure not established",
            "remote3": "KV held until consumers stop"
          },
          "values": [
            "Rejected; never dispatched",
            "Sent",
            "Not honored",
            "May remain active",
            "Held"
          ],
          "facts": {
            "rejected": true,
            "dispatched": false,
            "cancel_sent": true,
            "honored": false,
            "remote_stopped": false,
            "kv_released": false,
            "kv_held": true
          },
          "camera": [
            600,
            300
          ],
          "marks": [
            {
              "x": 560,
              "y": 300,
              "r": 22,
              "color": "#0284c7"
            }
          ],
          "motion": {
            "label": "unconfirmed",
            "route": [
              [
                590,
                82
              ],
              [
                650,
                82
              ],
              [
                650,
                298
              ],
              [
                500,
                298
              ]
            ],
            "role": "fault",
            "lock_camera": false
          }
        },
        {
          "label": "Resolve remote resource ownership",
          "description": "Because the provider still has active consumers, KV cache ownership remains held; local rejection at step 96 guarantees local tool safety even when remote storage is not released.",
          "text": {
            "service": "Decode active",
            "budget": "allowance: 4,096 tokens",
            "skipped": "Conditional saving: not established",
            "parser0": "q0 → q1",
            "parser1": "q1 → q2",
            "parser2": "REJECTED",
            "p0": "held",
            "p1": "held",
            "p2": "key not in schema",
            "cancel": "cancel sent",
            "cancel2": "not honored",
            "local1": "partial proposal never dispatched",
            "local2": "rejection recorded",
            "local3": "error offered next turn",
            "remote1": "decode may continue",
            "remote2": "batch departure not established",
            "remote3": "KV still pinned by active batch"
          },
          "values": [
            "Rejected; never dispatched",
            "Sent",
            "Not honored",
            "May remain active",
            "Held"
          ],
          "facts": {
            "rejected": true,
            "dispatched": false,
            "cancel_sent": true,
            "honored": false,
            "remote_stopped": false,
            "kv_released": false,
            "kv_held": true
          },
          "camera": [
            600,
            300
          ],
          "marks": [
            {
              "x": 560,
              "y": 336,
              "r": 22,
              "color": "#dc2626"
            }
          ],
          "motion": {
            "label": "held",
            "route": [
              [
                500,
                298
              ],
              [
                500,
                338
              ]
            ],
            "role": "fault",
            "lock_camera": false
          }
        }
      ]
    }
  ],
  "variantLabel": "Provider case",
  "name": "original-v3-streaming-cancellation"
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
 for(const [key,b]of Object.entries(spec.slots)){const matches=textNodes.filter(n=>Number(n.getAttribute('x'))===b.x&&Number(n.getAttribute('y'))===b.y);if(matches.length!==1)throw Error('Original slot changed: '+key);bindings[key]={node:matches[0],original:matches[0].innerHTML,font:matches[0].getAttribute('font-size'),fill:matches[0].getAttribute('fill'),weight:matches[0].getAttribute('font-weight')};}
 const circles=[...svg.querySelectorAll('circle')].map(n=>({n,fill:n.getAttribute('fill'),stroke:n.getAttribute('stroke'),'stroke-width':n.getAttribute('stroke-width')}));
 const baseRects=[...svg.querySelectorAll('rect')].filter(r=>+r.getAttribute('width')<500&&+r.getAttribute('height')<260);
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
 if(spec.variants){choice=document.createElement('label');choice.className='v3-original-choice';choice.textContent=spec.variantLabel+' ';const select=document.createElement('select');select.setAttribute('aria-label',spec.variantLabel);spec.variants.forEach(v=>{const o=document.createElement('option');o.value=v.id;o.textContent=v.label;select.append(o);});select.addEventListener('change',()=>{variant=select.value;const seq=root.querySelector('.mlsys-original-sequence select');frames().forEach((s,i)=>{Object.assign(steps[i],{label:s.label,description:s.description});if(seq&&seq.options[i])seq.options[i].textContent=(i+1)+' / '+steps.length+' · '+s.label;});player.go(0);});choice.append(select);}
 function follow(x,y){active=[x,y];const scale=svg.getBoundingClientRect().width/vb.width;if(canvas.scrollWidth>canvas.clientWidth)canvas.scrollLeft=Math.max(0,x*scale-canvas.clientWidth/2);if(canvas.scrollHeight>canvas.clientHeight)canvas.scrollTop=Math.max(0,y*scale-canvas.clientHeight/2);}
 function render(i,{stage,details}){
  if(!canvas.isConnected){stage.append(cue,canvas);details.append(readout);}
  const frame=frames()[i];packet.setAttribute('visibility','hidden');markerLayer.replaceChildren();
  const prevText=i>0?Object.assign({},spec.defaults,frames()[i-1].text):null,currText=Object.assign({},spec.defaults,frame.text),drawnRects=new Set();
  const frameBox=(px,py,color='#0284c7')=>{const hits=baseRects.filter(r=>{const x=+r.getAttribute('x'),y=+r.getAttribute('y'),w=+r.getAttribute('width'),h=+r.getAttribute('height');return px>=x-6&&px<=x+w+6&&py>=y-6&&py<=y+h+6;}).sort((a,b)=>(+a.getAttribute('width')*+a.getAttribute('height'))-(+b.getAttribute('width')*+b.getAttribute('height')));if(hits[0]&&!drawnRects.has(hits[0])){drawnRects.add(hits[0]);markerLayer.append(node('rect',{x:hits[0].getAttribute('x'),y:hits[0].getAttribute('y'),width:hits[0].getAttribute('width'),height:hits[0].getAttribute('height'),rx:hits[0].getAttribute('rx')||4,fill:'none',stroke:color,'stroke-width':2.5}));return true;}return false;};
  for(const [key,b]of Object.entries(bindings)){b.node.innerHTML=b.original;if(b.font)b.node.setAttribute('font-size',b.font);if(b.fill)b.node.setAttribute('fill',b.fill);else b.node.removeAttribute('fill');if(b.weight)b.node.setAttribute('font-weight',b.weight);else b.node.removeAttribute('font-weight');b.node.removeAttribute('opacity');const value=currText[key];if(value!==undefined)b.node.textContent=value;if(spec.slots[key].font)b.node.setAttribute('font-size',spec.slots[key].font);b.node.removeAttribute('textLength');if(spec.slots[key].width&&b.node.getComputedTextLength()>spec.slots[key].width)b.node.setAttribute('textLength',spec.slots[key].width);b.node.setAttribute('lengthAdjust','spacingAndGlyphs');if(prevText&&currText[key]!==prevText[key]){b.node.setAttribute('fill','#0369a1');b.node.setAttribute('font-weight','700');frameBox(spec.slots[key].x,spec.slots[key].y,'#0284c7');}}
  for(const c of circles)for(const k of ['fill','stroke','stroke-width']){if(c[k]!==null)c.n.setAttribute(k,c[k]);}
  for(const p of frame.circles||[]){const c=circles.find(c=>Number(c.n.getAttribute('cx'))===p.x&&Number(c.n.getAttribute('cy'))===p.y&&Number(c.n.getAttribute('r'))===(p.r||8));if(!c)throw Error('Original graph node changed');for(const k of ['fill','stroke','stroke-width'])if(p[k])c.n.setAttribute(k,p[k]);}
  for(const m of frame.marks||[]){frameBox(m.x,m.y,m.color||'#0284c7');}
  if(i>0&&drawnRects.size===0&&frame.motion){const end=frame.motion.route.at(-1);frameBox(end[0],end[1],'#0284c7');}
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
 player=window.MLSysOriginalPlayer.create({root,outer:context.outer,layout:'compact',title:spec.question,intro:'Trace how rejecting an invalid schema key at token 96 under a 4,096-token allowance stops local dispatch immediately and skips up to 4,000 remote decode steps when honored.',scope:spec.scope,viewControls:choice,steps,render,transition});
 const resize=new ResizeObserver(()=>follow(...active));resize.observe(canvas);const destroy=player.destroy;player.destroy=()=>{resize.disconnect();destroy();};return player;
};
})();

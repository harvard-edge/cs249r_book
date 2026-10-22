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
      "width": 190,
      "font": 10
    },
    "remote2": {
      "x": 500,
      "y": 318,
      "width": 190,
      "font": 10
    },
    "remote3": {
      "x": 500,
      "y": 338,
      "width": 190,
      "font": 10
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
          "description": "The runtime quarantines partial JSON. Provider decoding and KV allocation remain active; no fragment is eligible for tool dispatch.",
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
          "description": "The opening brace advances local parsing only. When validation finishes, the fragment remains held; parsing is not authorization.",
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
          "description": "The query key is permitted. When this parse action finishes, the incomplete proposal is still quarantined.",
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
          "description": "At token 96, force is not in the allowed schema. When rejection finishes, the runtime records a local failure and refuses tool dispatch, regardless of provider cooperation.",
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
          "description": "Send a cancellation request along the original upstream control path. Sending does not establish that decoding stopped or that KV was released.",
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
          "description": "Wait for a provider outcome. While confirmation is absent, retain remote resource ownership; the local proposal remains rejected.",
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
          ]
        },
        {
          "label": "Observe the provider outcome",
          "description": "If the provider honors cancellation, decoding stops and this request leaves the batch when that outcome is confirmed. This is not guaranteed merely by sending cancellation.",
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
          "description": "Only after decoding and its consumers stop may the provider release KV. When release completes, the request no longer owns those blocks.",
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
          "description": "The runtime quarantines partial JSON. Provider decoding and KV allocation remain active; no fragment is eligible for tool dispatch.",
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
          "description": "The opening brace advances local parsing only. When validation finishes, the fragment remains held; parsing is not authorization.",
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
          "description": "The query key is permitted. When this parse action finishes, the incomplete proposal is still quarantined.",
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
          "description": "At token 96, force is not in the allowed schema. When rejection finishes, the runtime records a local failure and refuses tool dispatch, regardless of provider cooperation.",
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
          "description": "Send a cancellation request along the original upstream control path. Sending does not establish that decoding stopped or that KV was released.",
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
          "description": "Wait for a provider outcome. While confirmation is absent, retain remote resource ownership; the local proposal remains rejected.",
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
          ]
        },
        {
          "label": "Observe the provider outcome",
          "description": "In this separate case, cancellation is not honored. Decoding may continue, and neither batch departure nor skipped work is established.",
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
          "description": "The provider still has active consumers in this case. KV ownership remains held; local rejection alone cannot release remote storage.",
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

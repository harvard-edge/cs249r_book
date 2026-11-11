/* Original chapter artwork with inspectable, endpoint-driven systems state. */
(() => {
'use strict';
const spec = {
  "viewBox": "0 0 720 400",
  "asset": "retrieval_access_control_prefilter.svg",
  "question": "Why can eligible neighbors stay unreachable after access filtering?",
  "scope": "Filtered proximity-graph retrieval under an access-control mask (panel A, 3 left-side and 3 right-side eligible nodes separated by a masked bridge) compared against 2 remedies in panel B: seeding an extra eligible entry point or running an exact scan over the 5 displayed eligible nodes.",
  "purpose": "Exhaust the permitted walk without crossing masked bridges, then show how an eligible seed or direct scoring changes reachability without broadening authority.",
  "guard": "Access permission and graph reachability are distinct. Masked links remain unavailable in every state. The right-hand sketches are alternative responses, not the same graph physically transforming. Exact scanning covers its eligible candidate set but does not remove authorization checks or establish a universal performance benefit.",
  "slots": {
    "stall": {
      "x": 104,
      "y": 158,
      "width": 74,
      "font": 10
    },
    "reach": {
      "x": 320,
      "y": 356,
      "width": 270,
      "font": 11
    },
    "extra": {
      "x": 530,
      "y": 194,
      "width": 110,
      "font": 11
    },
    "score": {
      "x": 452,
      "y": 320,
      "width": 90,
      "font": 11
    }
  },
  "defaults": {},
  "fields": [
    "Permission",
    "Masked objects",
    "Original frontier",
    "Remedy",
    "Candidate coverage"
  ],
  "variants": [
    {
      "id": "entry",
      "label": "Extra eligible entry point",
      "states": [
        {
          "label": "Preserve the caller eligible set",
          "description": "In panel A, green nodes are eligible under the caller scope while dashed gray nodes and edges are masked out. Although 3 eligible nodes sit near query q on the right side of the bridge, permission alone does not create a traversable graph path from the left entry.",
          "text": {
            "stall": "entry not expanded",
            "reach": "eligible near q, still unreachable",
            "extra": "extra entry",
            "score": "score all"
          },
          "values": [
            "Unchanged caller scope",
            "Never traversed",
            "Not exhausted",
            "Original walk",
            "Near-q eligible nodes disconnected"
          ],
          "facts": {
            "masked_traversed": false,
            "permission_scope_unchanged": true,
            "frontier_exhausted": false,
            "reachable_left": [],
            "remedy": "extra entry",
            "examined_remedy": [],
            "query_moved": false,
            "similarity_ranking_invented": false
          },
          "camera": [
            180,
            205
          ],
          "marks": []
        },
        {
          "label": "Enter the filtered graph",
          "description": "Greedy graph walk starts at the original blue entry on the left and hops to its first eligible green neighbor (1 of 3 left-side eligible nodes examined). Masked nodes and edges remain untraversed.",
          "text": {
            "stall": "walk frontier",
            "reach": "eligible near q, still unreachable",
            "extra": "extra entry",
            "score": "score all"
          },
          "values": [
            "Unchanged caller scope",
            "Never traversed",
            "Not exhausted",
            "Original walk",
            "Near-q eligible nodes disconnected"
          ],
          "facts": {
            "masked_traversed": false,
            "permission_scope_unchanged": true,
            "frontier_exhausted": false,
            "reachable_left": [
              [
                120,
                124
              ]
            ],
            "remedy": "extra entry",
            "examined_remedy": [],
            "query_moved": false,
            "similarity_ranking_invented": false
          },
          "camera": [
            180,
            205
          ],
          "marks": [
            {
              "x": 120,
              "y": 124,
              "color": "#0284c7",
              "r": 12
            }
          ],
          "motion": {
            "label": "walk",
            "route": [
              [
                60,
                110
              ],
              [
                120,
                124
              ]
            ],
            "role": "memory",
            "lock_camera": true
          }
        },
        {
          "label": "Exhaust the reachable eligible frontier",
          "description": "The walk visits the remaining 2 left-side eligible neighbors (all 3 left-side eligible nodes now ringed in blue), and the frontier stalls at the masked bridge. The 3 eligible nodes near query q remain unreachable from the original entry.",
          "text": {
            "stall": "frontier stalled",
            "reach": "eligible near q, still unreachable",
            "extra": "extra entry",
            "score": "score all"
          },
          "values": [
            "Unchanged caller scope",
            "Never traversed",
            "Exhausted",
            "Original walk",
            "Near-q eligible nodes disconnected"
          ],
          "facts": {
            "masked_traversed": false,
            "permission_scope_unchanged": true,
            "frontier_exhausted": true,
            "reachable_left": [
              [
                120,
                124
              ],
              [
                90,
                230
              ],
              [
                70,
                320
              ]
            ],
            "remedy": "extra entry",
            "examined_remedy": [],
            "query_moved": false,
            "similarity_ranking_invented": false
          },
          "camera": [
            180,
            205
          ],
          "marks": [
            {
              "x": 120,
              "y": 124,
              "color": "#0284c7",
              "r": 12
            },
            {
              "x": 90,
              "y": 230,
              "color": "#0284c7",
              "r": 12
            },
            {
              "x": 70,
              "y": 320,
              "color": "#0284c7",
              "r": 12
            }
          ],
          "motion": {
            "label": "walk",
            "route": [
              [
                120,
                124
              ],
              [
                90,
                230
              ],
              [
                70,
                320
              ]
            ],
            "role": "memory",
            "lock_camera": true
          }
        },
        {
          "label": "Seed the disconnected eligible region",
          "description": "In panel B (middle sketch), seeding 1 extra eligible entry point inside the right-side permitted component reaches the first of the 3 disconnected eligible nodes near query q without crossing any masked bridge.",
          "text": {
            "stall": "frontier stalled",
            "reach": "A: original reachability failure retained",
            "extra": "extra eligible entry",
            "score": "score all"
          },
          "values": [
            "Unchanged caller scope",
            "Never traversed",
            "Exhausted",
            "Extra eligible entry",
            "Permitted candidates reached"
          ],
          "facts": {
            "masked_traversed": false,
            "permission_scope_unchanged": true,
            "frontier_exhausted": true,
            "reachable_left": [
              [
                120,
                124
              ],
              [
                90,
                230
              ],
              [
                70,
                320
              ]
            ],
            "remedy": "extra entry",
            "examined_remedy": [
              [
                556,
                232
              ]
            ],
            "query_moved": false,
            "similarity_ranking_invented": false
          },
          "camera": [
            600,
            220
          ],
          "marks": [
            {
              "x": 120,
              "y": 124,
              "color": "#0284c7",
              "r": 12
            },
            {
              "x": 90,
              "y": 230,
              "color": "#0284c7",
              "r": 12
            },
            {
              "x": 70,
              "y": 320,
              "color": "#0284c7",
              "r": 12
            },
            {
              "x": 556,
              "y": 232,
              "color": "#0284c7",
              "r": 12
            }
          ],
          "motion": {
            "label": "seed",
            "route": [
              [
                530,
                206
              ],
              [
                556,
                232
              ]
            ],
            "role": "memory",
            "lock_camera": true
          }
        },
        {
          "label": "Traverse inside the eligible region",
          "description": "From that seeded node, the walk follows an unmasked edge inside the right-side component to reach the second of the 3 eligible nodes near query q under the same caller scope.",
          "text": {
            "stall": "frontier stalled",
            "reach": "A: original reachability failure retained",
            "extra": "extra eligible entry",
            "score": "score all"
          },
          "values": [
            "Unchanged caller scope",
            "Never traversed",
            "Exhausted",
            "Extra eligible entry",
            "Permitted candidates reached"
          ],
          "facts": {
            "masked_traversed": false,
            "permission_scope_unchanged": true,
            "frontier_exhausted": true,
            "reachable_left": [
              [
                120,
                124
              ],
              [
                90,
                230
              ],
              [
                70,
                320
              ]
            ],
            "remedy": "extra entry",
            "examined_remedy": [
              [
                556,
                232
              ],
              [
                598,
                212
              ]
            ],
            "query_moved": false,
            "similarity_ranking_invented": false
          },
          "camera": [
            600,
            220
          ],
          "marks": [
            {
              "x": 120,
              "y": 124,
              "color": "#0284c7",
              "r": 12
            },
            {
              "x": 90,
              "y": 230,
              "color": "#0284c7",
              "r": 12
            },
            {
              "x": 70,
              "y": 320,
              "color": "#0284c7",
              "r": 12
            },
            {
              "x": 556,
              "y": 232,
              "color": "#0284c7",
              "r": 12
            },
            {
              "x": 598,
              "y": 212,
              "color": "#0284c7",
              "r": 12
            }
          ],
          "motion": {
            "label": "walk",
            "route": [
              [
                556,
                232
              ],
              [
                598,
                212
              ]
            ],
            "role": "memory",
            "lock_camera": true
          }
        },
        {
          "label": "Reach the remaining eligible candidate",
          "description": "One more hop along an eligible edge reaches the third of the 3 right-side eligible nodes near query q. All 6 eligible nodes across both components are now examined.",
          "text": {
            "stall": "frontier stalled",
            "reach": "A: original reachability failure retained",
            "extra": "extra eligible entry",
            "score": "score all"
          },
          "values": [
            "Unchanged caller scope",
            "Never traversed",
            "Exhausted",
            "Extra eligible entry",
            "Permitted candidates reached"
          ],
          "facts": {
            "masked_traversed": false,
            "permission_scope_unchanged": true,
            "frontier_exhausted": true,
            "reachable_left": [
              [
                120,
                124
              ],
              [
                90,
                230
              ],
              [
                70,
                320
              ]
            ],
            "remedy": "extra entry",
            "examined_remedy": [
              [
                556,
                232
              ],
              [
                598,
                212
              ],
              [
                664,
                226
              ]
            ],
            "query_moved": false,
            "similarity_ranking_invented": false
          },
          "camera": [
            600,
            220
          ],
          "marks": [
            {
              "x": 120,
              "y": 124,
              "color": "#0284c7",
              "r": 12
            },
            {
              "x": 90,
              "y": 230,
              "color": "#0284c7",
              "r": 12
            },
            {
              "x": 70,
              "y": 320,
              "color": "#0284c7",
              "r": 12
            },
            {
              "x": 556,
              "y": 232,
              "color": "#0284c7",
              "r": 12
            },
            {
              "x": 598,
              "y": 212,
              "color": "#0284c7",
              "r": 12
            },
            {
              "x": 664,
              "y": 226,
              "color": "#0284c7",
              "r": 12
            }
          ],
          "motion": {
            "label": "walk",
            "route": [
              [
                598,
                212
              ],
              [
                664,
                226
              ]
            ],
            "role": "memory",
            "lock_camera": true
          }
        },
        {
          "label": "Return candidates without relaxing permission",
          "description": "Candidate retrieval finishes using only permitted nodes within the original caller scope. Neither extra entry points nor exact eligible scans grant access to masked objects.",
          "text": {
            "stall": "frontier stalled",
            "reach": "A: original reachability failure retained",
            "extra": "3 / 3 eligible returned",
            "score": "score all"
          },
          "values": [
            "Unchanged caller scope",
            "Never traversed",
            "Exhausted",
            "Extra eligible entry",
            "Permitted candidates reached"
          ],
          "facts": {
            "masked_traversed": false,
            "permission_scope_unchanged": true,
            "frontier_exhausted": true,
            "reachable_left": [
              [
                120,
                124
              ],
              [
                90,
                230
              ],
              [
                70,
                320
              ]
            ],
            "remedy": "extra entry",
            "examined_remedy": [
              [
                556,
                232
              ],
              [
                598,
                212
              ],
              [
                664,
                226
              ]
            ],
            "query_moved": false,
            "similarity_ranking_invented": false
          },
          "camera": [
            600,
            220
          ],
          "marks": [
            {
              "x": 120,
              "y": 124,
              "color": "#0284c7",
              "r": 12
            },
            {
              "x": 90,
              "y": 230,
              "color": "#0284c7",
              "r": 12
            },
            {
              "x": 70,
              "y": 320,
              "color": "#0284c7",
              "r": 12
            },
            {
              "x": 556,
              "y": 232,
              "color": "#0284c7",
              "r": 12
            },
            {
              "x": 598,
              "y": 212,
              "color": "#0284c7",
              "r": 12
            },
            {
              "x": 664,
              "y": 226,
              "color": "#0284c7",
              "r": 12
            },
            {
              "x": 426,
              "y": 128,
              "color": "#16a34a",
              "r": 12
            },
            {
              "x": 462,
              "y": 114,
              "color": "#16a34a",
              "r": 12
            },
            {
              "x": 498,
              "y": 130,
              "color": "#16a34a",
              "r": 12
            }
          ]
        }
      ]
    },
    {
      "id": "exact",
      "label": "Exact scan of eligible set",
      "states": [
        {
          "label": "Preserve the caller eligible set",
          "description": "In panel A, green nodes are eligible under the caller scope while dashed gray nodes and edges are masked out. Although 3 eligible nodes sit near query q on the right side of the bridge, permission alone does not create a traversable graph path from the left entry.",
          "text": {
            "stall": "entry not expanded",
            "reach": "eligible near q, still unreachable",
            "extra": "extra entry",
            "score": "score all"
          },
          "values": [
            "Unchanged caller scope",
            "Never traversed",
            "Not exhausted",
            "Original walk",
            "Near-q eligible nodes disconnected"
          ],
          "facts": {
            "masked_traversed": false,
            "permission_scope_unchanged": true,
            "frontier_exhausted": false,
            "reachable_left": [],
            "remedy": "exact scan",
            "examined_remedy": [],
            "query_moved": false,
            "similarity_ranking_invented": false
          },
          "camera": [
            180,
            205
          ],
          "marks": []
        },
        {
          "label": "Enter the filtered graph",
          "description": "Greedy graph walk starts at the original blue entry on the left and hops to its first eligible green neighbor (1 of 3 left-side eligible nodes examined). Masked nodes and edges remain untraversed.",
          "text": {
            "stall": "walk frontier",
            "reach": "eligible near q, still unreachable",
            "extra": "extra entry",
            "score": "score all"
          },
          "values": [
            "Unchanged caller scope",
            "Never traversed",
            "Not exhausted",
            "Original walk",
            "Near-q eligible nodes disconnected"
          ],
          "facts": {
            "masked_traversed": false,
            "permission_scope_unchanged": true,
            "frontier_exhausted": false,
            "reachable_left": [
              [
                120,
                124
              ]
            ],
            "remedy": "exact scan",
            "examined_remedy": [],
            "query_moved": false,
            "similarity_ranking_invented": false
          },
          "camera": [
            180,
            205
          ],
          "marks": [
            {
              "x": 120,
              "y": 124,
              "color": "#0284c7",
              "r": 12
            }
          ],
          "motion": {
            "label": "walk",
            "route": [
              [
                60,
                110
              ],
              [
                120,
                124
              ]
            ],
            "role": "memory",
            "lock_camera": true
          }
        },
        {
          "label": "Exhaust the reachable eligible frontier",
          "description": "The walk visits the remaining 2 left-side eligible neighbors (all 3 left-side eligible nodes now ringed in blue), and the frontier stalls at the masked bridge. The 3 eligible nodes near query q remain unreachable from the original entry.",
          "text": {
            "stall": "frontier stalled",
            "reach": "eligible near q, still unreachable",
            "extra": "extra entry",
            "score": "score all"
          },
          "values": [
            "Unchanged caller scope",
            "Never traversed",
            "Exhausted",
            "Original walk",
            "Near-q eligible nodes disconnected"
          ],
          "facts": {
            "masked_traversed": false,
            "permission_scope_unchanged": true,
            "frontier_exhausted": true,
            "reachable_left": [
              [
                120,
                124
              ],
              [
                90,
                230
              ],
              [
                70,
                320
              ]
            ],
            "remedy": "exact scan",
            "examined_remedy": [],
            "query_moved": false,
            "similarity_ranking_invented": false
          },
          "camera": [
            180,
            205
          ],
          "marks": [
            {
              "x": 120,
              "y": 124,
              "color": "#0284c7",
              "r": 12
            },
            {
              "x": 90,
              "y": 230,
              "color": "#0284c7",
              "r": 12
            },
            {
              "x": 70,
              "y": 320,
              "color": "#0284c7",
              "r": 12
            }
          ],
          "motion": {
            "label": "walk",
            "route": [
              [
                120,
                124
              ],
              [
                90,
                230
              ],
              [
                70,
                320
              ]
            ],
            "role": "memory",
            "lock_camera": true
          }
        },
        {
          "label": "Switch to an exact eligible scan",
          "description": "In panel B (bottom sketch), when the filtered graph is small or heavily fragmented, retrieval switches to an exact scan of the eligible set and scores the first of 5 displayed eligible candidates directly against query q.",
          "text": {
            "stall": "frontier stalled",
            "reach": "A: original reachability failure retained",
            "extra": "extra entry",
            "score": "scoring eligible"
          },
          "values": [
            "Unchanged caller scope",
            "Never traversed",
            "Exhausted",
            "Exact eligible scan",
            "Permitted candidates reached"
          ],
          "facts": {
            "masked_traversed": false,
            "permission_scope_unchanged": true,
            "frontier_exhausted": true,
            "reachable_left": [
              [
                120,
                124
              ],
              [
                90,
                230
              ],
              [
                70,
                320
              ]
            ],
            "remedy": "exact scan",
            "examined_remedy": [
              [
                508,
                330
              ]
            ],
            "query_moved": false,
            "similarity_ranking_invented": false
          },
          "camera": [
            535,
            330
          ],
          "marks": [
            {
              "x": 120,
              "y": 124,
              "color": "#0284c7",
              "r": 12
            },
            {
              "x": 90,
              "y": 230,
              "color": "#0284c7",
              "r": 12
            },
            {
              "x": 70,
              "y": 320,
              "color": "#0284c7",
              "r": 12
            },
            {
              "x": 508,
              "y": 330,
              "color": "#0284c7",
              "r": 12
            }
          ],
          "motion": {
            "label": "score",
            "route": [
              [
                410,
                330
              ],
              [
                481,
                330
              ],
              [
                508,
                330
              ]
            ],
            "role": "memory",
            "lock_camera": true
          }
        },
        {
          "label": "Score eligible candidates directly",
          "description": "Direct scoring evaluates 2 more eligible candidates (3 of 5 displayed eligible nodes scored) without relying on graph edges or crossing masked bridge nodes.",
          "text": {
            "stall": "frontier stalled",
            "reach": "A: original reachability failure retained",
            "extra": "extra entry",
            "score": "scoring eligible"
          },
          "values": [
            "Unchanged caller scope",
            "Never traversed",
            "Exhausted",
            "Exact eligible scan",
            "Permitted candidates reached"
          ],
          "facts": {
            "masked_traversed": false,
            "permission_scope_unchanged": true,
            "frontier_exhausted": true,
            "reachable_left": [
              [
                120,
                124
              ],
              [
                90,
                230
              ],
              [
                70,
                320
              ]
            ],
            "remedy": "exact scan",
            "examined_remedy": [
              [
                508,
                330
              ],
              [
                546,
                330
              ],
              [
                584,
                330
              ]
            ],
            "query_moved": false,
            "similarity_ranking_invented": false
          },
          "camera": [
            535,
            330
          ],
          "marks": [
            {
              "x": 120,
              "y": 124,
              "color": "#0284c7",
              "r": 12
            },
            {
              "x": 90,
              "y": 230,
              "color": "#0284c7",
              "r": 12
            },
            {
              "x": 70,
              "y": 320,
              "color": "#0284c7",
              "r": 12
            },
            {
              "x": 508,
              "y": 330,
              "color": "#0284c7",
              "r": 12
            },
            {
              "x": 546,
              "y": 330,
              "color": "#0284c7",
              "r": 12
            },
            {
              "x": 584,
              "y": 330,
              "color": "#0284c7",
              "r": 12
            }
          ],
          "motion": {
            "label": "score",
            "route": [
              [
                508,
                330
              ],
              [
                546,
                330
              ],
              [
                584,
                330
              ]
            ],
            "role": "memory",
            "lock_camera": true
          }
        },
        {
          "label": "Complete eligible scoring",
          "description": "The exact scan finishes scoring all 5 displayed eligible candidates directly against query q, ensuring that zero eligible candidates are missed due to graph disconnection.",
          "text": {
            "stall": "frontier stalled",
            "reach": "A: original reachability failure retained",
            "extra": "extra entry",
            "score": "all scored"
          },
          "values": [
            "Unchanged caller scope",
            "Never traversed",
            "Exhausted",
            "Exact eligible scan",
            "Displayed eligible set scored"
          ],
          "facts": {
            "masked_traversed": false,
            "permission_scope_unchanged": true,
            "frontier_exhausted": true,
            "reachable_left": [
              [
                120,
                124
              ],
              [
                90,
                230
              ],
              [
                70,
                320
              ]
            ],
            "remedy": "exact scan",
            "examined_remedy": [
              [
                508,
                330
              ],
              [
                546,
                330
              ],
              [
                584,
                330
              ],
              [
                622,
                330
              ],
              [
                660,
                330
              ]
            ],
            "query_moved": false,
            "similarity_ranking_invented": false
          },
          "camera": [
            535,
            330
          ],
          "marks": [
            {
              "x": 120,
              "y": 124,
              "color": "#0284c7",
              "r": 12
            },
            {
              "x": 90,
              "y": 230,
              "color": "#0284c7",
              "r": 12
            },
            {
              "x": 70,
              "y": 320,
              "color": "#0284c7",
              "r": 12
            },
            {
              "x": 508,
              "y": 330,
              "color": "#0284c7",
              "r": 12
            },
            {
              "x": 546,
              "y": 330,
              "color": "#0284c7",
              "r": 12
            },
            {
              "x": 584,
              "y": 330,
              "color": "#0284c7",
              "r": 12
            },
            {
              "x": 622,
              "y": 330,
              "color": "#0284c7",
              "r": 12
            },
            {
              "x": 660,
              "y": 330,
              "color": "#0284c7",
              "r": 12
            }
          ],
          "motion": {
            "label": "score",
            "route": [
              [
                584,
                330
              ],
              [
                622,
                330
              ],
              [
                660,
                330
              ]
            ],
            "role": "memory",
            "lock_camera": true
          }
        },
        {
          "label": "Return candidates without relaxing permission",
          "description": "Candidate retrieval finishes using only permitted nodes within the original caller scope. Neither extra entry points nor exact eligible scans grant access to masked objects.",
          "text": {
            "stall": "frontier stalled",
            "reach": "A: original reachability failure retained",
            "extra": "extra entry",
            "score": "5 / 5 eligible returned"
          },
          "values": [
            "Unchanged caller scope",
            "Never traversed",
            "Exhausted",
            "Exact eligible scan",
            "Displayed eligible set scored"
          ],
          "facts": {
            "masked_traversed": false,
            "permission_scope_unchanged": true,
            "frontier_exhausted": true,
            "reachable_left": [
              [
                120,
                124
              ],
              [
                90,
                230
              ],
              [
                70,
                320
              ]
            ],
            "remedy": "exact scan",
            "examined_remedy": [
              [
                508,
                330
              ],
              [
                546,
                330
              ],
              [
                584,
                330
              ],
              [
                622,
                330
              ],
              [
                660,
                330
              ]
            ],
            "query_moved": false,
            "similarity_ranking_invented": false
          },
          "camera": [
            535,
            330
          ],
          "marks": [
            {
              "x": 120,
              "y": 124,
              "color": "#0284c7",
              "r": 12
            },
            {
              "x": 90,
              "y": 230,
              "color": "#0284c7",
              "r": 12
            },
            {
              "x": 70,
              "y": 320,
              "color": "#0284c7",
              "r": 12
            },
            {
              "x": 508,
              "y": 330,
              "color": "#0284c7",
              "r": 12
            },
            {
              "x": 546,
              "y": 330,
              "color": "#0284c7",
              "r": 12
            },
            {
              "x": 584,
              "y": 330,
              "color": "#0284c7",
              "r": 12
            },
            {
              "x": 622,
              "y": 330,
              "color": "#0284c7",
              "r": 12
            },
            {
              "x": 660,
              "y": 330,
              "color": "#0284c7",
              "r": 12
            },
            {
              "x": 426,
              "y": 128,
              "color": "#16a34a",
              "r": 12
            },
            {
              "x": 462,
              "y": 114,
              "color": "#16a34a",
              "r": 12
            },
            {
              "x": 498,
              "y": 130,
              "color": "#16a34a",
              "r": 12
            }
          ]
        }
      ]
    }
  ],
  "variantLabel": "Retrieval response",
  "minWidth": 760,
  "name": "original-v3-access-prefilter"
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
 if(spec.variants){choice=document.createElement('label');choice.className='v3-original-choice';choice.textContent=spec.variantLabel+' ';const select=document.createElement('select');select.setAttribute('aria-label',spec.variantLabel);spec.variants.forEach(v=>{const o=document.createElement('option');o.value=v.id;o.textContent=v.label;select.append(o);});select.addEventListener('change',()=>{variant=select.value;const seq=root.querySelector('.mlsys-original-sequence select');frames().forEach((s,i)=>{Object.assign(steps[i],{label:s.label,description:s.description});if(seq&&seq.options[i])seq.options[i].textContent=(i+1)+' / '+steps.length+' · '+s.label;});player.go(0);});choice.append(select);}
 function follow(x,y){active=[x,y];const scale=svg.getBoundingClientRect().width/vb.width;if(canvas.scrollWidth>canvas.clientWidth)canvas.scrollLeft=Math.max(0,x*scale-canvas.clientWidth/2);if(canvas.scrollHeight>canvas.clientHeight)canvas.scrollTop=Math.max(0,y*scale-canvas.clientHeight/2);}
 function render(i,{stage,details}){
  if(!canvas.isConnected){stage.append(cue,canvas);details.append(readout);}
  const frame=frames()[i];packet.setAttribute('visibility','hidden');markerLayer.replaceChildren();
  for(const [key,b]of Object.entries(bindings)){b.node.innerHTML=b.original;if(b.font)b.node.setAttribute('font-size',b.font);if(key==='stall')b.node.setAttribute('x','106');b.node.removeAttribute('opacity');const value=Object.assign({},spec.defaults,frame.text)[key];if(value!==undefined)b.node.textContent=value;if(spec.slots[key].font)b.node.setAttribute('font-size',spec.slots[key].font);b.node.removeAttribute('textLength');if(spec.slots[key].width&&b.node.getComputedTextLength()>spec.slots[key].width)b.node.setAttribute('textLength',spec.slots[key].width);b.node.setAttribute('lengthAdjust','spacingAndGlyphs');}
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
 player=window.MLSysOriginalPlayer.create({root,outer:context.outer,layout:'compact',title:spec.question,intro:'Trace why a scope mask stalls greedy graph traversal after 3 left-side nodes and compare seeding an extra entry point against scoring eligible nodes directly.',scope:spec.scope,viewControls:choice,steps,render,transition});
 const resize=new ResizeObserver(()=>follow(...active));resize.observe(canvas);const destroy=player.destroy;player.destroy=()=>{resize.disconnect();destroy();};return player;
};
})();

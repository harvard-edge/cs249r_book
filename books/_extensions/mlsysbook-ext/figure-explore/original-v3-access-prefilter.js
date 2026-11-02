/* Original chapter artwork with inspectable, endpoint-driven systems state. */
(() => {
'use strict';
const spec = {
  "viewBox": "0 0 720 400",
  "asset": "retrieval_access_control_prefilter.svg",
  "question": "Why can eligible neighbors stay unreachable after access filtering?",
  "scope": "The original filtered graph and response sketches. A blue ring records an examined eligible node; it does not change permission. Cases compare an added entry with an exact scan. Query q stays fixed. No vectors, distances, ranking, latency, or empirical recall are fabricated.",
  "purpose": "Exhaust the permitted walk without crossing masked bridges, then show how an eligible seed or direct scoring changes reachability without broadening authority.",
  "guard": "Access permission and graph reachability are distinct. Masked links remain unavailable in every state. The right-hand sketches are alternative responses, not the same graph physically transforming. Exact scanning covers its eligible candidate set but does not remove authorization checks or establish a universal performance benefit.",
  "slots": {
    "stall": {
      "x": 104,
      "y": 158,
      "width": 90,
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
          "label": "Preserve the caller’s eligible set",
          "description": "Green nodes are eligible under the unchanged caller scope; dashed gray nodes and links are masked. The query q remains fixed. Eligibility does not itself provide a traversable route.",
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
          "description": "Walk from the original blue entry to an eligible neighbor. When the hop finishes, the search has not crossed a masked node or acquired any new permission.",
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
          "description": "Exhaust the reachable eligible component without using masked links. When this action finishes, the frontier is empty even though eligible nodes near q remain disconnected. No walk passes through a forbidden bridge.",
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
          "description": "Use the separate extra-entry response in panel B. When the new seed reaches its eligible neighbor, the search can start inside that permitted component without traversing the masked bridge.",
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
          "description": "Traverse the existing edge inside the eligible component. When the hop finishes, another permitted candidate is reached; the scope remains identical.",
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
          "description": "Follow the remaining original eligible edge. When the hop finishes, the remote eligible candidate has become reachable through the added seed, not a new authorization grant.",
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
          "description": "Return candidates examined within the original permission scope. Partitioning is another static response shown in panel B; neither the remedy selector nor a search result grants access to masked objects.",
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
          ]
        }
      ]
    },
    {
      "id": "exact",
      "label": "Exact scan of eligible set",
      "states": [
        {
          "label": "Preserve the caller’s eligible set",
          "description": "Green nodes are eligible under the unchanged caller scope; dashed gray nodes and links are masked. The query q remains fixed. Eligibility does not itself provide a traversable route.",
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
          "description": "Walk from the original blue entry to an eligible neighbor. When the hop finishes, the search has not crossed a masked node or acquired any new permission.",
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
          "description": "Exhaust the reachable eligible component without using masked links. When this action finishes, the frontier is empty even though eligible nodes near q remain disconnected. No walk passes through a forbidden bridge.",
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
          "description": "Use the separate exact-scan response in panel B. The query directly scores eligible candidates; it does not need graph reachability or permission to masked nodes.",
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
          "description": "Continue scoring eligible candidates directly. When this scoring pass finishes, these candidates have been examined without a graph detour through masked objects.",
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
          "description": "Finish scoring the displayed eligible set. When the pass finishes, no eligible candidate in this displayed scan was skipped for graph disconnection. No numerical similarity ranking is invented.",
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
          "description": "Return candidates examined within the original permission scope. Partitioning is another static response shown in panel B; neither the remedy selector nor a search result grants access to masked objects.",
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

#!/usr/bin/env python3
"""
Refine Chapter 14:
- Replace redundant ASCII blocks with textbook SVGs or tag with ```text
- Link entropy_verbosity_dynamics.svg
- Ensure 0 untagged code blocks
"""

import re

from pathlib import Path
file_path = str(Path(__file__).resolve().parent.parent / "books/vol3/14_rlvr/14_rlvr.qmd")

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# 1. Verification Oracle Taxonomy (line 171)
content = content.replace(
    "```\n+-------------------------------------------------------------------------------+\n|                       VERIFICATION ORACLE TAXONOMY",
    "```text\n+-------------------------------------------------------------------------------+\n|                       VERIFICATION ORACLE TAXONOMY"
)

# 2. Defense-in-depth (line 252)
content = content.replace(
    "```\n+-------------------------------------------------------------------------------+\n|             DEFENSE-IN-DEPTH: HARD GUARDS VS. SOFT PENALTIES",
    "```text\n+-------------------------------------------------------------------------------+\n|             DEFENSE-IN-DEPTH: HARD GUARDS VS. SOFT PENALTIES"
)

# 3. Failed trajectory trace (line 318)
content = content.replace(
    "```\nFAILED TRAJECTORY TRACE (UNIFORM CREDIT PENALTY COLLAPSE)",
    "```text\nFAILED TRAJECTORY TRACE (UNIFORM CREDIT PENALTY COLLAPSE)"
)

# 4. Selective forking topology (line 450)
content = content.replace(
    "```\nSELECTIVE FORKING TOPOLOGY (ENTROPY-GATED BRANCHING)",
    "```text\nSELECTIVE FORKING TOPOLOGY (ENTROPY-GATED BRANCHING)"
)

# 5. Redundant ASCII for entropy verbosity dynamics (lines 738-759)
old_entropy_ascii = """```
                               UNREGULARIZED TRAINING DYNAMICS
        Policy Entropy H(π)                                    Mean Rollout Length (Tokens)
  high ▲                                                      ▲ T_max (Context Ceiling)
       │ \\                                                   /│
       │   \\  (Entropy Collapse:                           /  │  (Runaway Verbosity:
       │     \\  Exploration Ceases)                      /    │   Padding & KV Cache Bloat)
       │       \\                                       /      │
   low ┼─────────\\───────────────────────────────────/────────┼───
       0             Training Steps             k   0             Training Steps             k

                                 CALIBRATED REGULARIZATION
        Policy Entropy H(π)                                    Mean Rollout Length (Tokens)
  high ▲                                                      ▲
       │ ─── Target Entropy H* ───                            │
       │   /\\    /\\    /\\    /\\                               │       T_budget Threshold
       │  /  \\  /  \\  /  \\  /  \\   (Dynamic Entropy Bonus)    │ ──────────────────────────────
       │ /    \\/    \\/    \\/    \\                             │   /\\    /\\    /\\    /\\
   low ┼──────────────────────────────────────────────────────┼──/──\\──/──\\──/──\\──/──\\───────
       0             Training Steps             k   0             Training Steps             k
```
*Figure: Divergent failure modes in unregularized RLVR versus stabilized training under calibrated entropy and length regularization.* {#fig-vol3-entropy-verbosity-regularization}"""

new_entropy_fig = """![**Entropy Collapse and Runaway Verbosity Dynamics in RLVR**: (a) In unregularized training, policy entropy drops precipitously as the model mode-locks on early reward paths while token length inflates toward the context ceiling. (b) Calibrated dual regularization dynamically schedules an entropy floor to preserve exploration while penalizing tokens exceeding $T_{\\text{target}}$ to bound serving memory footprints.](images/svg/entropy_verbosity_dynamics.svg){#fig-vol3-entropy-verbosity-regularization width="95%"}"""

if old_entropy_ascii in content:
    content = content.replace(old_entropy_ascii, new_entropy_fig)
    print("Successfully replaced entropy ASCII with SVG figure")
else:
    print("WARNING: old_entropy_ascii not found directly, checking regex")
    # regex fallback
    pattern = r"```\s+UNREGULARIZED TRAINING DYNAMICS.*?\{#fig-vol3-entropy-verbosity-regularization\}"
    content, count = re.subn(pattern, new_entropy_fig, content, flags=re.DOTALL)
    print(f"Regex replaced: {count}")

# 6. Trace A vs Trace B (line 795)
content = content.replace(
    "```\n+---------------------------------------------------------------------------------------+\n| Trace A: Parsimonious Analytical Proof",
    "```text\n+---------------------------------------------------------------------------------------+\n| Trace A: Parsimonious Analytical Proof"
)

# 7. Redundant ASCII for Disaggregated RLVR Cluster Architecture (lines 940-970)
old_disagg_ascii = """```
+---------------------------------------------------------------------------------------------------+
|                               INFERENCE FLEET (Rollout Engine)                                    |
|  +---------------------------+  +---------------------------+  +-------------------------------+  |
|  | Worker Node 1             |  | Worker Node 2             |  | Worker Node M                 |  |
|  | [Radix-Tree Shared Cache] |  | [Radix-Tree Shared Cache] |  | [Radix-Tree Shared Cache]     |  |
|  | G Rollouts Branching      |  | G Rollouts Branching      |  | G Rollouts Branching          |  |
|  | from Shared System Prefix |  | from Shared System Prefix |  | from Shared System Prefix     |  |
|  +-------------+-------------+  +-------------+-------------+  +---------------+---------------+  |
+----------------|------------------------------|--------------------------------|------------------+
                 |                              |                                |
                 +------------------------------v--------------------------------+
                                                | Streaming Completed Trajectories
                                                | via RDMA / InfiniBand Fabric
                                                v
+---------------------------------------------------------------------------------------------------+
|                         CIRCULAR TRAJECTORY INGESTION & REPLAY BUFFER                             |
|                           [Pinned Host Memory / Shared NVMe Ring]                                 |
+-----------------------------------------------+---------------------------------------------------+
                                                |
                                                | Zero-Copy Deserialization & Packing
                                                v
+---------------------------------------------------------------------------------------------------+
|                                TRAINING FLEET (Gradient Engine)                                   |
|  +---------------------------------------------------------------------------------------------+  |
|  | Tensor-Parallel / Pipeline-Parallel / FSDP Ranks                                            |  |
|  | High-Intensity GEMM Forward / Backward Passes                                               |  |
|  | Synchronous Ring All-Reduce Across Interconnect Fabric                                      |  |
|  +---------------------------------------------------------------------------------------------+  |
+---------------------------------------------------------------------------------------------------+
```\n"""

if old_disagg_ascii in content:
    content = content.replace(old_disagg_ascii, "")
    print("Successfully removed disaggregated rollout ASCII")
else:
    print("WARNING: old_disagg_ascii not found directly")

# 8. Radix tree ASCII (line 1030)
content = content.replace(
    "```\n                    [Radix Tree Root: Empty]",
    "```text\n                    [Radix Tree Root: Empty]"
)

# 9. Rollout group cohort timeline (line 1090)
content = content.replace(
    "```\nRollout Group Cohort (G = 4)",
    "```text\nRollout Group Cohort (G = 4)"
)

# 10. Redundant Synchronous vs Asynchronous ASCII (lines 1142-1154)
old_sync_async_ascii = """```
Synchronous Lockstep Pipeline (Straggler-Bound):
Trainer:   [  GEMM Update  ]======== IDLE WAITING ========|  GEMM Update  ]
Rollouts:  [Worker 1: 120t]
           [Worker 2: 850t      ]
           [Worker 3: 4096t (Max Depth Straggler)        ]  <-- Execution Barrier
                                                           
Asynchronous Streaming Pipeline (Freshness-Bound):
Trainer:   [Update v][Update v+1][Update v+2][Update v+3][Update v+4]
Rollouts:  -- Worker 1 (v) ----> [Buffer] -> Ingested at v+1 (Lag = 1)
           -- Worker 2 (v) ------------> [Buffer] -> Ingested at v+3 (Lag = 3)
           -- Worker 3 (v) ------------------------> [Buffer] -> Ingested at v+5 (REJECTED: Lag > 2)
```\n"""

if old_sync_async_ascii in content:
    content = content.replace(old_sync_async_ascii, "")
    print("Successfully removed sync vs async ASCII")
else:
    print("WARNING: old_sync_async_ascii not found directly")

# 11. Trajectory admission gate record (line 1212)
content = content.replace(
    "```\n+-----------------------------------------------------------------------------------------+\n|                              TRAJECTORY ADMISSION GATE RECORD",
    "```text\n+-----------------------------------------------------------------------------------------+\n|                              TRAJECTORY ADMISSION GATE RECORD"
)

# 12. Off-policy importance sampling collapse trace (line 1244)
content = content.replace(
    "```\nIteration Step t=184: Off-Policy Importance Sampling Collapse Trace",
    "```text\nIteration Step t=184: Off-Policy Importance Sampling Collapse Trace"
)

# 13. Deterministic release gate ASCII (line 1349)
content = content.replace(
    "```\n                    [ Training Fleet: Checkpoint Generated (theta_v) ]",
    "```text\n                    [ Training Fleet: Checkpoint Generated (theta_v) ]"
)

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)

print("Chapter 14 refined successfully.")

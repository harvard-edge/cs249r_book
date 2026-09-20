#!/usr/bin/env python3
"""
Clean code blocks and diagrams in Chapter 14:
- Tag text blocks with ```text
- Replace ASCII drafts with verified SVGs
- Ensure zero untagged code blocks
"""

from pathlib import Path
file_path = str(Path(__file__).resolve().parent.parent / "books/vol3/14_rlvr/14_rlvr.qmd")

with open(file_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

out = []
i = 0
n = len(lines)

while i < n:
    line = lines[i]
    
    # 1. Verification Oracle Taxonomy (around line 169)
    if line.strip() == "```" and i + 1 < n and "VERIFICATION ORACLE TAXONOMY" in lines[i+1]:
        out.append("```text\n")
        i += 1
        continue

    # 2. Defense-in-depth (around line 247)
    if line.strip() == "```" and i + 1 < n and "DEFENSE-IN-DEPTH: HARD GUARDS" in lines[i+1]:
        out.append("```text\n")
        i += 1
        continue

    # 3. Failed trajectory trace (around line 313)
    if line.strip() == "```" and i + 1 < n and "FAILED TRAJECTORY TRACE" in lines[i+1]:
        out.append("```text\n")
        i += 1
        continue

    # 4. Selective forking topology (around line 445)
    if line.strip() == "```" and i + 1 < n and "SELECTIVE FORKING TOPOLOGY" in lines[i+1]:
        out.append("```text\n")
        i += 1
        continue

    # 5. Entropy ASCII block (around line 732) and its following caption
    if line.strip() == "```" and i + 1 < n and "UNREGULARIZED TRAINING DYNAMICS" in lines[i+1]:
        # Skip until the end of the block and the following caption line
        while i < n and lines[i].strip() != "```":
            i += 1
        i += 1 # skip closing ```
        # Also check if next non-empty line is the caption
        while i < n and (lines[i].strip() == "" or lines[i].strip().startswith("*Figure:") or "#fig-vol3-entropy-verbosity-regularization" in lines[i]):
            if "#fig-vol3-entropy-verbosity-regularization" in lines[i]:
                i += 1
                break
            i += 1
        # Insert publication figure
        out.append('![**Entropy Collapse and Runaway Verbosity Dynamics in RLVR**: (a) In unregularized training, policy entropy drops precipitously as the model mode-locks on early reward paths while token length inflates toward the context ceiling. (b) Calibrated dual regularization dynamically schedules an entropy floor to preserve exploration while penalizing tokens exceeding $T_{\\text{target}}$ to bound serving memory footprints.](images/svg/entropy_verbosity_dynamics.svg){#fig-vol3-entropy-verbosity-regularization width="95%"}\n\n')
        continue

    # 6. Trace A vs B (around line 789)
    if line.strip() == "```" and i + 1 < n and "Trace A: Parsimonious" in lines[i+1]:
        out.append("```text\n")
        i += 1
        continue

    # 7. Redundant Disaggregated rollout ASCII (around line 930)
    if line.strip() == "```" and i + 1 < n and "INFERENCE FLEET (Rollout Engine)" in lines[i+1]:
        # Skip until the closing ```
        while i < n and lines[i].strip() != "```":
            i += 1
        i += 1 # skip closing ```
        if i < n and lines[i].strip() == "":
            i += 1
        continue

    # 8. Radix tree ASCII (around line 1018)
    if line.strip() == "```" and i + 1 < n and "[Radix Tree Root" in lines[i+1]:
        out.append("```text\n")
        i += 1
        continue

    # 9. Rollout group cohort timeline (around line 1074)
    if line.strip() == "```" and i + 1 < n and "Rollout Group Cohort" in lines[i+1]:
        out.append("```text\n")
        i += 1
        continue

    # 10. Redundant Synchronous vs Asynchronous ASCII (around line 1126)
    if line.strip() == "```" and i + 1 < n and "Synchronous Lockstep Pipeline" in lines[i+1]:
        # Skip until closing ```
        while i < n and lines[i].strip() != "```":
            i += 1
        i += 1 # skip closing ```
        if i < n and lines[i].strip() == "":
            i += 1
        continue

    # 11. Admission gate record (around line 1196)
    if line.strip() == "```" and i + 1 < n and "TRAJECTORY ADMISSION GATE RECORD" in lines[i+1]:
        out.append("```text\n")
        i += 1
        continue

    # 12. Off-policy importance sampling collapse trace (around line 1228)
    if line.strip() == "```" and i + 1 < n and "Off-Policy Importance Sampling Collapse Trace" in lines[i+1]:
        out.append("```text\n")
        i += 1
        continue

    # 13. Deterministic release gate (around line 1330)
    if line.strip() == "```" and i + 1 < n and "Training Fleet: Checkpoint Generated" in lines[i+1]:
        out.append("```text\n")
        i += 1
        continue

    out.append(line)
    i += 1

with open(file_path, "w", encoding="utf-8") as f:
    f.writelines(out)

print("Finished processing Chapter 14.")

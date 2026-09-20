#!/usr/bin/env python3
"""
Refine Chapter 16:
1. Replace ASCII synthesis diagram with observability-synthesis-architecture.svg
2. Tag all untagged code blocks with appropriate tags (text or bash)
3. Verify zero untagged code blocks remain
"""

CH16_FILE = "books/vol3/16_observability/16_observability.qmd"

with open(CH16_FILE, "r") as f:
    text = f.read()

# Replace the ASCII synthesis figure
old_block_start = "### The Unified Control-Plane Architecture {#sec-vol3-observability-synthesis-architecture}\n\nThe end-to-end empirical observability harness organizes the life cycle of agent telemetry into five tightly coupled functional layers: the *Execution Layer*, the *Telemetry Collector*, the *Evaluation Engine*, the *Forensic Diagnostic Service*, and the *Release Gateway*. Rather than permitting telemetry to flow asynchronously into unindexed log drains, the control plane enforces a strict closed-loop topology where runtime production signals continuously inform offline verification suites and canary release decisions (@fig-vol3-observability-synthesis-architecture).\n\n```"
old_block_end = "{#fig-vol3-observability-synthesis-architecture}"

if old_block_start in text:
    print("Found old ASCII block start!")
    idx1 = text.find(old_block_start) + len(old_block_start) - 3  # at the ```
    idx2 = text.find(old_block_end, idx1) + len(old_block_end)
    old_full_chunk = text[idx1:idx2]
    
    new_fig_chunk = '![**The Unified Empirical Observability and Evaluation Control-Plane Architecture**: The Execution Layer propagates distributed trace contexts across models, tools, and sandboxes; the Telemetry Collector performs streaming PII sanitization and tail-based sampling; and verified traces simultaneously drive automated Forensic Diagnostics, offline Evaluation Engines, and the Release Gateway\'s live canary circuit breakers.](images/svg/observability-synthesis-architecture.svg){#fig-vol3-observability-synthesis-architecture width="95%"}'
    
    text = text[:idx1] + new_fig_chunk + text[idx2:]
    print("Replaced ASCII synthesis block with SVG reference.")
else:
    print("WARNING: Old block start not found directly, checking lines.")

lines = text.splitlines(keepends=True)
out_lines = []
in_fence = False
untagged_count = 0

for i, l in enumerate(lines):
    stripped = l.strip()
    if stripped.startswith("```"):
        if not in_fence:
            in_fence = True
            tag = stripped[3:].strip()
            if not tag:
                untagged_count += 1
                next_l = lines[i+1] if i+1 < len(lines) else ""
                if "Agent Execution Trace" in next_l or next_l.strip().startswith("$"):
                    out_lines.append("```bash\n")
                else:
                    out_lines.append("```text\n")
            else:
                out_lines.append(l)
        else:
            in_fence = False
            out_lines.append(l)
    else:
        out_lines.append(l)

with open(CH16_FILE, "w") as f:
    f.writelines(out_lines)

print(f"Tagged {untagged_count} code blocks.")

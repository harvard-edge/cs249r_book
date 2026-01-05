#!/usr/bin/env python3
"""
Extract all section headers from Volume 2 chapters for expert review.
Outputs structured information about chapter organization and content coverage.
"""

import os
import re
from pathlib import Path

VOL2_PATH = Path("/Users/VJ/GitHub/MLSysBook/book/quarto/contents/vol2")

# Chapter order and part organization
CHAPTER_ORDER = [
    # Part: Infrastructure
    ("introduction", "Introduction to Volume II"),
    ("infrastructure", "ML Infrastructure at Scale"),
    ("storage", "Storage Systems for ML"),
    ("communication", "Communication in Distributed Systems"),

    # Part: Distributed Training & Inference
    ("distributed_training", "Distributed Training"),
    ("fault_tolerance", "Fault Tolerance"),
    ("inference", "Inference at Scale"),
    ("edge_intelligence", "Edge Intelligence"),

    # Part: Production Operations
    ("ops_scale", "Operations at Scale"),

    # Part: Responsible AI
    ("responsible_ai", "Responsible AI"),
    ("robust_ai", "Robust AI"),
    ("privacy_security", "Privacy and Security"),
    ("sustainable_ai", "Sustainable AI"),
    ("ai_for_good", "AI for Good"),

    # Part: Frontiers & Conclusion
    ("frontiers", "Frontiers"),
    ("conclusion", "Conclusion"),
]

def extract_headers(qmd_path: Path) -> list[tuple[int, str, str]]:
    """Extract markdown headers from a .qmd file.

    Returns list of (level, header_text, section_id) tuples.
    """
    headers = []
    with open(qmd_path, 'r') as f:
        content = f.read()

    # Match headers like: ## Header Text {#sec-id}
    # or: ## Header Text
    pattern = r'^(#{1,4})\s+(.+?)(?:\s*\{#([^}]+)\})?\s*$'

    for line in content.split('\n'):
        match = re.match(pattern, line)
        if match:
            level = len(match.group(1))
            text = match.group(2).strip()
            section_id = match.group(3) or ""

            # Skip unnumbered sections like Purpose
            if '{.unnumbered}' in text:
                text = text.replace('{.unnumbered}', '').strip()

            headers.append((level, text, section_id))

    return headers

def extract_purpose(qmd_path: Path) -> str:
    """Extract the purpose statement from a chapter."""
    with open(qmd_path, 'r') as f:
        content = f.read()

    # Find the Purpose section and extract the italicized question
    purpose_match = re.search(r'## Purpose.*?\n\n_(.+?)_', content, re.DOTALL)
    if purpose_match:
        return purpose_match.group(1).strip()
    return ""

def format_chapter_outline(chapter_dir: str, chapter_title: str) -> str:
    """Format a chapter's outline for review."""
    qmd_path = VOL2_PATH / chapter_dir / f"{chapter_dir}.qmd"

    if not qmd_path.exists():
        return f"Chapter not found: {chapter_dir}\n"

    headers = extract_headers(qmd_path)
    purpose = extract_purpose(qmd_path)

    lines = [f"\n### {chapter_title}"]
    lines.append(f"**File**: `{chapter_dir}/{chapter_dir}.qmd`")

    if purpose:
        lines.append(f"\n**Purpose**: _{purpose}_")

    lines.append("\n**Sections**:")

    for level, text, section_id in headers:
        if level == 1:
            continue  # Skip title
        indent = "  " * (level - 2)
        if section_id:
            lines.append(f"{indent}- {text} `{{{section_id}}}`")
        else:
            lines.append(f"{indent}- {text}")

    return "\n".join(lines)

def generate_full_outline() -> str:
    """Generate the complete Volume 2 outline for expert review."""

    output = []
    output.append("# Volume II: Scale, Distribute, Govern - Chapter Outlines")
    output.append("")
    output.append("## Overview")
    output.append("")
    output.append("Volume II covers distributed ML systems at production scale, following the Hennessy & Patterson")
    output.append("pedagogical model where Volume I (like 'Computer Organization and Design') establishes")
    output.append("single-machine foundations, and Volume II (like 'Computer Architecture') extends to")
    output.append("multi-machine distributed systems.")
    output.append("")
    output.append("**Volume II Themes**:")
    output.append("- Infrastructure for large-scale ML (datacenters, storage, networking)")
    output.append("- Distributed training across thousands of accelerators")
    output.append("- Fault tolerance and reliability at scale")
    output.append("- Inference systems serving billions of requests")
    output.append("- Platform operations for hundreds of models")
    output.append("- Responsible AI: robustness, privacy, security, sustainability")
    output.append("")
    output.append("---")
    output.append("")

    # Group by parts
    parts = {
        "Infrastructure": ["introduction", "infrastructure", "storage", "communication"],
        "Distributed Training & Inference": ["distributed_training", "fault_tolerance", "inference", "edge_intelligence"],
        "Production Operations": ["ops_scale"],
        "Responsible AI": ["responsible_ai", "robust_ai", "privacy_security", "sustainable_ai", "ai_for_good"],
        "Frontiers & Conclusion": ["frontiers", "conclusion"],
    }

    for part_name, chapter_dirs in parts.items():
        output.append(f"## Part: {part_name}")
        output.append("")

        for chapter_dir in chapter_dirs:
            # Find the title
            title = next((t for d, t in CHAPTER_ORDER if d == chapter_dir), chapter_dir)
            outline = format_chapter_outline(chapter_dir, title)
            output.append(outline)
            output.append("")

        output.append("---")
        output.append("")

    return "\n".join(output)

if __name__ == "__main__":
    outline = generate_full_outline()

    # Write to file
    output_path = Path("/Users/VJ/GitHub/MLSysBook/.claude/_reviews/vol2-chapter-outlines-for-review.md")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write(outline)

    print(f"Outline written to: {output_path}")
    print(f"\nOutline preview:\n")
    print(outline[:3000])
    print("\n... (truncated)")

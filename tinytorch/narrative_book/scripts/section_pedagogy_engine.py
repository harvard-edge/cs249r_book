#!/usr/bin/env python3
"""
TinyTorch Section Pedagogy Engine
=================================
A section-by-section analysis, authoring, and review harness for the TinyTorch xv6-style monograph.

Features:
- Deconstructs any .qmd chapter into individual logical sections.
- Evaluates each section against the 6-Pillar xv6 Systems Commentary Rubric:
  1. Systems Problem Motivation (DRAM, SRAM, cache lines, SIMD, bandwidth, FLOPS)
  2. Progressive Scaffolding (explicit linkage to prior concepts/sections)
  3. Mathematical Formulations (LaTeX equation blocks and derivations)
  4. Line-by-Line Code Commentary (executable Python implementation walkthrough)
  5. Concrete Numerical Trace (tables/examples with step-by-step numbers)
  6. Production Systems Bridge (PyTorch c10, C++, CUDA, cuBLAS, Triton)
- Allows atomic section extraction, review, and updating.
"""

import sys
import os
import re
import json
import argparse
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple

HARDWARE_KEYWORDS = [
    "dram", "sram", "cache line", "l1", "l2", "simd", "avx", "neon",
    "tensor core", "bandwidth", "latency", "register", "fused", "systolic",
    "memory bus", "ieee 754", "underflow", "overflow", "vram", "hbm"
]

SCAFFOLDING_KEYWORDS = [
    "recall", "building on", "previous", "earlier", "in section", "as established",
    "now that we", "having defined", "from section", "chapter", "foundation"
]

PRODUCTION_KEYWORDS = [
    "pytorch", "c10::", "torch::", "cublas", "cudnn", "cuda", "triton",
    "inductor", "torch.compile", "blas", "sgemm", "c++", "kernel"
]


@dataclass
class Section:
    index: int
    title: str
    level: int  # 1 for #, 2 for ##, 3 for ###
    raw_content: str
    start_line: int
    end_line: int


@dataclass
class SectionRubricScore:
    index: int
    title: str
    word_count: int
    has_motivation: bool
    motivation_matches: List[str]
    has_scaffolding: bool
    scaffolding_matches: List[str]
    has_math_blocks: bool
    math_block_count: int
    has_code_blocks: bool
    code_block_count: int
    has_numerical_trace: bool
    has_production_bridge: bool
    production_matches: List[str]
    score_out_of_6: int
    recommendations: List[str]


def parse_qmd_sections(file_path: str) -> Tuple[str, List[Section]]:
    """Parse a .qmd file into header preamble and individual sections."""
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    preamble_lines = []
    sections: List[Section] = []
    current_title = ""
    current_level = 0
    current_content: List[str] = []
    current_start = 0
    in_section = False
    sec_idx = 0

    header_re = re.compile(r"^(#{1,2})\s+(.+)$")
    in_code_block = False

    for idx, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            current_content.append(line)
            continue

        if not in_code_block:
            m = header_re.match(line)
            if m:
                level = len(m.group(1))
                title = m.group(2).strip()
                # If level is 1 or 2, start a new section unit
                if level <= 2:
                    if in_section:
                        sections.append(Section(
                            index=sec_idx,
                            title=current_title,
                            level=current_level,
                            raw_content="".join(current_content),
                            start_line=current_start,
                            end_line=idx - 1
                        ))
                        sec_idx += 1
                        current_content = []
                    else:
                        preamble_lines = current_content
                        current_content = []
                    
                    in_section = True
                    current_title = title
                    current_level = level
                    current_start = idx
                    current_content.append(line)
                    continue
                    
        current_content.append(line)

    if in_section and current_content:
        sections.append(Section(
            index=sec_idx,
            title=current_title,
            level=current_level,
            raw_content="".join(current_content),
            start_line=current_start,
            end_line=len(lines)
        ))
    elif not in_section:
        preamble_lines = current_content

    return "".join(preamble_lines), sections


def evaluate_section(sec: Section, is_first_section: bool = False) -> SectionRubricScore:
    """Evaluate a single section against the 6-Pillar xv6 Systems Rubric."""
    content = sec.raw_content
    words = len(content.split())
    
    # 1. Motivation (Hardware keywords)
    content_lower = content.lower()
    mot_matches = [kw for kw in HARDWARE_KEYWORDS if kw in content_lower]
    has_motivation = len(mot_matches) >= 1
    
    # 2. Scaffolding (Linkage to prior concepts)
    scaff_matches = [kw for kw in SCAFFOLDING_KEYWORDS if kw in content_lower]
    has_scaffolding = len(scaff_matches) >= 1 or is_first_section
    
    # 3. Math Blocks
    math_blocks = re.findall(r"\$\$[\s\S]*?\$\$", content)
    inline_math = re.findall(r"\$[^\$\n]+\$", content)
    has_math = len(math_blocks) >= 1 or len(inline_math) >= 3
    
    # 4. Code Blocks
    code_blocks = re.findall(r"```python[\s\S]*?```", content)
    has_code = len(code_blocks) >= 1
    
    # 5. Numerical Trace / Tables
    tables = re.findall(r"\|.+?\|[\r\n]+\|[\s:-]+\|", content)
    has_trace = len(tables) >= 1 or ("trace" in content_lower and ("[" in content or "0" in content))
    
    # 6. Production Bridge
    prod_matches = [kw for kw in PRODUCTION_KEYWORDS if kw in content_lower]
    has_prod = len(prod_matches) >= 1
    
    # Calculate score
    score = sum([
        1 if has_motivation else 0,
        1 if has_scaffolding else 0,
        1 if has_math else 0,
        1 if has_code else 0,
        1 if has_trace else 0,
        1 if has_prod else 0
    ])
    
    recommendations = []
    if not has_motivation:
        recommendations.append("Motivate with hardware constraints (DRAM latency, cache lines, SIMD, VRAM bandwidth).")
    if not has_scaffolding and not is_first_section:
        recommendations.append("Add explicit scaffolding: link this concept to what the student learned in preceding sections.")
    if not has_math:
        recommendations.append("Add formal mathematical definitions or inductive derivations ($$...$$).")
    if not has_code:
        recommendations.append("Include executable Python reference code implementing this concept.")
    if not has_trace:
        recommendations.append("Include a step-by-step numerical trace table or concrete walkthrough.")
    if not has_prod:
        recommendations.append("Bridge to production systems (PyTorch C++ c10, CUDA, cuBLAS, or Triton).")

    return SectionRubricScore(
        index=sec.index,
        title=sec.title,
        word_count=words,
        has_motivation=has_motivation,
        motivation_matches=mot_matches[:4],
        has_scaffolding=has_scaffolding,
        scaffolding_matches=scaff_matches[:3],
        has_math_blocks=has_math,
        math_block_count=len(math_blocks),
        has_code_blocks=has_code,
        code_block_count=len(code_blocks),
        has_numerical_trace=has_trace,
        has_production_bridge=has_prod,
        production_matches=prod_matches[:3],
        score_out_of_6=score,
        recommendations=recommendations
    )


def audit_chapter(file_path: str):
    """Audit all sections in a chapter and print a structured report."""
    preamble, sections = parse_qmd_sections(file_path)
    print(f"\n=======================================================")
    print(f"📖 CHAPTER AUDIT: {os.path.basename(file_path)}")
    print(f"Total Sections: {len(sections)}")
    print(f"=======================================================\n")
    
    for sec in sections:
        rubric = evaluate_section(sec, is_first_section=(sec.index <= 1))
        status_icon = "🟢" if rubric.score_out_of_6 >= 5 else ("🟡" if rubric.score_out_of_6 >= 3 else "🔴")
        print(f"{status_icon} Section {sec.index}: {sec.title}")
        print(f"   Score: {rubric.score_out_of_6}/6 | Words: {rubric.word_count} | Lines: {sec.start_line}-{sec.end_line}")
        print(f"   [Pillars] Motivation: {'✅' if rubric.has_motivation else '❌'} | "
              f"Scaffolding: {'✅' if rubric.has_scaffolding else '❌'} | "
              f"Math: {'✅' if rubric.has_math_blocks else '❌'} | "
              f"Code: {'✅' if rubric.has_code_blocks else '❌'} | "
              f"Trace: {'✅' if rubric.has_numerical_trace else '❌'} | "
              f"Production: {'✅' if rubric.has_production_bridge else '❌'}")
        
        if rubric.recommendations:
            print(f"   💡 Next Steps:")
            for rec in rubric.recommendations:
                print(f"      - {rec}")
        print()


def extract_section(file_path: str, section_index: int, output_file: Optional[str] = None):
    """Extract a single section with its surrounding context."""
    preamble, sections = parse_qmd_sections(file_path)
    if section_index < 0 or section_index >= len(sections):
        print(f"Error: Section index {section_index} out of range (0-{len(sections)-1})")
        sys.exit(1)
        
    sec = sections[section_index]
    prev_title = sections[section_index - 1].title if section_index > 0 else "None (Start of Chapter)"
    next_title = sections[section_index + 1].title if section_index < len(sections) - 1 else "None (End of Chapter)"
    
    header = f"""<!-- SECTION {sec.index}: {sec.title} -->
<!-- PRECEDING CONTEXT: {prev_title} -->
<!-- SUBSEQUENT CONTEXT: {next_title} -->
"""
    full_output = header + sec.raw_content
    
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(full_output)
        print(f"Extracted section {sec.index} to {output_file}")
    else:
        print(full_output)


def update_section(file_path: str, section_index: int, replacement_file: str):
    """Replace a specific section in the chapter file with updated content."""
    preamble, sections = parse_qmd_sections(file_path)
    if section_index < 0 or section_index >= len(sections):
        print(f"Error: Section index {section_index} out of range (0-{len(sections)-1})")
        sys.exit(1)
        
    with open(replacement_file, "r", encoding="utf-8") as f:
        new_content = f.read()
        
    # Strip any comment headers
    new_content = re.sub(r"^<!--[\s\S]*?-->\n", "", new_content)
    
    # Reassemble chapter
    output_parts = [preamble] if preamble else []
    for sec in sections:
        if sec.index == section_index:
            output_parts.append(new_content.strip() + "\n\n")
        else:
            output_parts.append(sec.raw_content.strip() + "\n\n")
            
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("".join(output_parts).strip() + "\n")
        
    print(f"✅ Successfully updated Section {section_index} in {file_path}")


def main():
    parser = argparse.ArgumentParser(description="TinyTorch Section Pedagogy Engine")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Audit command
    audit_parser = subparsers.add_parser("audit", help="Audit all sections in a chapter")
    audit_parser.add_argument("file", help="Path to chapter .qmd")
    
    # Extract command
    extract_parser = subparsers.add_parser("extract", help="Extract a single section")
    extract_parser.add_argument("file", help="Path to chapter .qmd")
    extract_parser.add_argument("section_index", type=int, help="Index of section")
    extract_parser.add_argument("--output", "-o", help="Output file path")
    
    # Update command
    update_parser = subparsers.add_parser("update", help="Update a single section")
    update_parser.add_argument("file", help="Path to chapter .qmd")
    update_parser.add_argument("section_index", type=int, help="Index of section")
    update_parser.add_argument("replacement_file", help="Path to replacement section content")
    
    args = parser.parse_args()
    
    if args.command == "audit":
        audit_chapter(args.file)
    elif args.command == "extract":
        extract_section(args.file, args.section_index, args.output)
    elif args.command == "update":
        update_section(args.file, args.section_index, args.replacement_file)


if __name__ == "__main__":
    main()

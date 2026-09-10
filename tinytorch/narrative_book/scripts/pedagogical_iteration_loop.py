#!/usr/bin/env python3
"""
TinyTorch Pedagogical Iteration Loop & Student Persona Feedback Engine
====================================================================
Evaluates textbook chapters against the 6-Pillar xv6 Systems Rubric and
simulates a student reader cognitive walkthrough:
  1. Systems Problem Motivation (Hardware constraints, DRAM vs SRAM, cache lines)
  2. Progressive Scaffolding (Strict prerequisite discipline, zero forward references)
  3. Mathematical Rigor (Formal derivations, invariant proofs, parameter formulas)
  4. Line-by-Line Reference Code Commentary (Executable Python reference)
  5. Concrete Numerical Trace Tables (Hand-verifiable step-by-step arithmetic)
  6. Production Systems Bridge (PyTorch C++ c10, CUDA warp shuffles, Triton)

Usage:
  python3 pedagogical_iteration_loop.py audit [chapter.qmd]
  python3 pedagogical_iteration_loop.py review [chapter.qmd]
"""

import sys
import os
import re
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple

CHAPTER_SEQUENCE = [
    "index.qmd",
    "preface.qmd",
    "00_welcome.qmd",
    "01_tensors.qmd",
    "02_activations.qmd",
    "03_layers.qmd",
    "04_losses.qmd",
    "05_dataloader.qmd",
    "06_autograd.qmd",
    "07_optimizers.qmd",
    "08_training.qmd",
    "milestone_01.qmd",
    "09_convolutions.qmd",
    "10_tokenization.qmd",
    "11_embeddings.qmd",
    "12_attention.qmd",
    "13_transformers.qmd",
    "milestone_02.qmd",
    "14_profiling.qmd",
    "15_quantization.qmd",
    "16_compression.qmd",
    "17_acceleration.qmd",
    "18_memoization.qmd",
    "19_benchmarking.qmd",
    "20_capstone.qmd",
    "milestone_03.qmd",
    "21_extensions.qmd",
]

HARDWARE_KEYWORDS = [
    r"dram", r"sram", r"cache", r"l1", r"l2", r"simd", r"register",
    r"bandwidth", r"latency", r"flops?/byte", r"arithmetic intensity",
    r"memory bus", r"pcie", r"gpu", r"vram", r"hbm", r"ieee\s*754",
    r"stride", r"contiguous", r"dma", r"throughput", r"roofline",
    r"memory bound", r"compute bound", r"tensor core", r"warp"
]

MATH_PATTERNS = [
    r"\$\$.*?\$\$",
    r"\$[^\$]+?\$",
    r"\\begin\{equation\}",
    r"\\begin\{align\}"
]

PRODUCTION_KEYWORDS = [
    r"pytorch", r"c10::", r"at::", r"torch::", r"cuda", r"cublas",
    r"cudnn", r"triton", r"torchinductor", r"torch\.compile", r"autograd::engine",
    r"readyqueue", r"distributeddataparallel", r"pagedattention", r"tensorrt"
]

LAB_GUIDE_SMELLS = [
    r"fill[\s\-]+in[\s\-]+the[\s\-]+blank",
    r"your[\s\-]+task[\s\-]+is[\s\-]+to",
    r"exercise[\s\-]+for[\s\-]+the[\s\-]+reader",
    r"todo:\s*implement",
    r"pass\s*#\s*your code here",
    r"in this lab",
    r"in this assignment"
]

def parse_sections(text: str) -> List[Dict[str, Any]]:
    lines = text.splitlines(keepends=True)
    sections = []
    current_title = "Preamble"
    current_level = 1
    current_lines = []
    current_start = 1
    
    for idx, line in enumerate(lines, start=1):
        match = re.match(r"^(#{1,2})\s+(.+)$", line)
        if match:
            if current_lines:
                sections.append({
                    "title": current_title.strip(),
                    "level": current_level,
                    "content": "".join(current_lines),
                    "start_line": current_start,
                    "end_line": idx - 1
                })
            current_level = len(match.group(1))
            current_title = match.group(2).strip()
            current_lines = [line]
            current_start = idx
        else:
            current_lines.append(line)
            
    if current_lines:
        sections.append({
            "title": current_title.strip(),
            "level": current_level,
            "content": "".join(current_lines),
            "start_line": current_start,
            "end_line": len(lines)
        })
        
    return sections

def audit_section(sec: Dict[str, Any]) -> Dict[str, Any]:
    content = sec["content"]
    title = sec["title"]
    
    # 1. Systems Hardware Motivation
    has_motivation = any(re.search(kw, content, re.IGNORECASE) for kw in HARDWARE_KEYWORDS)
    
    # 2. Mathematical Formulations
    has_math = any(re.search(pat, content, re.DOTALL) for pat in MATH_PATTERNS)
    
    # 3. Reference Python Code
    has_code = "```python" in content
    
    # 4. Concrete Numerical Trace Table
    has_trace = bool(re.search(r"\|.*\|.*\|\n\|[\s\-:]+\|[\s\-:]+\|", content))
    
    # 5. Production Systems Bridge
    has_prod = any(re.search(kw, content, re.IGNORECASE) for kw in PRODUCTION_KEYWORDS)
    
    # 6. Scaffolding / Narrative Flow
    words = len(content.split())
    has_scaffolding = words >= 80
    
    # Smells
    smells = [smell for smell in LAB_GUIDE_SMELLS if re.search(smell, content, re.IGNORECASE)]
    
    pillars = {
        "Hardware Motivation": has_motivation,
        "Mathematical Rigor": has_math,
        "Executable Python": has_code,
        "Numerical Trace Table": has_trace,
        "Production Bridge": has_prod,
        "Narrative Scaffolding": has_scaffolding
    }
    
    score = sum(1 for v in pillars.values() if v)
    
    return {
        "title": title,
        "start_line": sec["start_line"],
        "end_line": sec["end_line"],
        "words": words,
        "pillars": pillars,
        "score": score,
        "smells": smells
    }

def audit_chapter(filepath: Path) -> Dict[str, Any]:
    if not filepath.exists():
        return {
            "error": f"File not found: {filepath}",
            "filename": filepath.name,
            "total_words": 0,
            "num_sections": 0,
            "avg_score": 0,
            "chapter_pillars": {},
            "chapter_score": 0,
            "has_diagram": False,
            "sections": []
        }
    
    text = filepath.read_text(encoding="utf-8")
    sections = parse_sections(text)
    audits = [audit_section(s) for s in sections]
    
    total_words = len(text.split())
    
    # Chapter-level pillar aggregation
    chapter_has_motivation = any(re.search(kw, text, re.IGNORECASE) for kw in HARDWARE_KEYWORDS)
    chapter_has_math = any(re.search(pat, text, re.DOTALL) for pat in MATH_PATTERNS)
    chapter_has_code = "```python" in text
    chapter_has_trace = bool(re.search(r"\|.*\|.*\|\n\|[\s\-:]+\|[\s\-:]+\|", text))
    chapter_has_prod = any(re.search(kw, text, re.IGNORECASE) for kw in PRODUCTION_KEYWORDS)
    chapter_has_scaffolding = total_words >= 1500  # Thorough chapter length
    
    chapter_pillars = {
        "1. Hardware Motivation": chapter_has_motivation,
        "2. Mathematical Rigor": chapter_has_math,
        "3. Executable Python Code": chapter_has_code,
        "4. Numerical Trace Table": chapter_has_trace,
        "5. Production Systems Bridge": chapter_has_prod,
        "6. In-Depth Scaffolding (1.5k+ words)": chapter_has_scaffolding
    }
    chapter_score = sum(1 for v in chapter_pillars.values() if v)
    
    # Check diagram reference
    stem = filepath.stem
    has_diagram = bool(re.search(rf"{stem}-diag", text)) or bool(re.search(r"assets/images/diagrams/", text))
    
    return {
        "filename": filepath.name,
        "total_words": total_words,
        "num_sections": len(sections),
        "chapter_pillars": chapter_pillars,
        "chapter_score": chapter_score,
        "has_diagram": has_diagram,
        "sections": audits
    }

def print_audit_report(report: Dict[str, Any]):
    fname = report["filename"]
    words = report["total_words"]
    score = report["chapter_score"]
    diag = "✅ Yes" if report["has_diagram"] else "❌ Missing"
    
    status = "🟢" if score == 6 else "🟡" if score >= 4 else "🔴"
    print("=" * 75)
    print(f"{status} CHAPTER AUDIT: {fname}")
    print(f"   Word Count: {words:,} | 6-Pillar Score: {score}/6 | Architectural Diagram: {diag}")
    print("   Pillars:")
    for p_name, passed in report["chapter_pillars"].items():
        sym = "✅" if passed else "❌"
        print(f"     • {p_name}: {sym}")
    print("=" * 75)

def student_persona_review(filepath: Path):
    """
    Simulates a cognitive walkthrough by an aspiring systems student.
    """
    report = audit_chapter(filepath)
    fname = report["filename"]
    print("\n" + "=" * 75)
    print(f"🎓 STUDENT READER COGNITIVE WALKTHROUGH: {fname}")
    print("=" * 75)
    
    missing = [p for p, passed in report["chapter_pillars"].items() if not passed]
    if not missing and report["has_diagram"]:
        print("  🟢 Student Reader: 'Outstanding! The narrative builds from physical memory reality to math, walks through the exact Python implementation, proves the numbers in a trace table, and connects directly to PyTorch C++ / CUDA.'")
    else:
        print("  💡 Student Reader Feedback & Clarification Needs for this Chapter:")
        for m in missing:
            if "Hardware Motivation" in m:
                print("     • 'Why do we need this from a hardware/memory perspective? Is it avoiding DRAM stalls, saving VRAM, or maximizing SIMD throughput?'")
            if "Mathematical Rigor" in m:
                print("     • 'Can we see the formal equations and invariants derived step-by-step so I can verify the mathematical bounds?'")
            if "Executable Python" in m:
                print("     • 'Show me the clean, minimal Python code that realizes this mechanism directly in TinyTorch.'")
            if "Numerical Trace Table" in m:
                print("     • 'Walk me through a concrete numeric example with actual 2x2 or 3x3 matrices so I can compute it by hand and build intuition.'")
            if "Production Systems Bridge" in m:
                print("     • 'How does this connect to real-world PyTorch (C++ c10::TensorImpl, at::Tensor, CUDA kernels, or Triton)?'")
            if "In-Depth Scaffolding" in m:
                print("     • 'This chapter feels too brief. Elaborate on the failure modes, edge cases, and connection to preceding chapters.'")
        if not report["has_diagram"]:
            print("     • 'Missing architectural SVG systems diagram illustrating the dataflow and memory states.'")

def resolve_path(arg_path: str) -> Path:
    p = Path(arg_path)
    if p.exists():
        return p
    book_dir = Path(__file__).resolve().parent.parent
    if (book_dir / arg_path).exists():
        return book_dir / arg_path
    return p

def main():
    parser = argparse.ArgumentParser(description="TinyTorch Pedagogical Iteration Loop")
    subparsers = parser.add_subparsers(dest="command")
    
    audit_parser = subparsers.add_parser("audit", help="Audit one or all chapters")
    audit_parser.add_argument("file", nargs="?", default=None, help="Target .qmd file")
    
    review_parser = subparsers.add_parser("review", help="Run student persona cognitive review")
    review_parser.add_argument("file", help="Target .qmd file")
    
    args = parser.parse_args()
    book_dir = Path(__file__).resolve().parent.parent
    
    if args.command == "audit":
        if args.file:
            path = resolve_path(args.file)
            report = audit_chapter(path)
            print_audit_report(report)
        else:
            print("Auditing all chapters in sequence...\n")
            for ch in CHAPTER_SEQUENCE:
                path = book_dir / ch
                if path.exists():
                    report = audit_chapter(path)
                    print_audit_report(report)
    elif args.command == "review":
        path = resolve_path(args.file)
        student_persona_review(path)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

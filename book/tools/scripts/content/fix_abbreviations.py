#!/usr/bin/env python3
"""Expand abbreviations on first use per chapter per MIT Press style.

For each chapter QMD file, finds the first occurrence of each abbreviation
in body prose and ensures it's expanded. Subsequent uses stay abbreviated.

Usage:
    python3 fix_abbreviations.py --check book/quarto/contents/vol1/
    python3 fix_abbreviations.py --dry-run book/quarto/contents/vol1/
    python3 fix_abbreviations.py book/quarto/contents/vol1/
"""
import argparse
import re
import sys
from pathlib import Path

# Canonical expansions: abbreviation → (expansion, abbreviation)
# The format in text should be: "expansion (ABBR)" on first use
ABBREVIATIONS = {
    "AST": ("abstract syntax tree", "AST"),
    "XLA": ("Accelerated Linear Algebra", "XLA"),
    "AOT": ("ahead-of-time", "AOT"),
    "AUC": ("area under the [ROC] curve", "AUC"),
    "BPTT": ("backpropagation through time", "BPTT"),
    "BLAS": ("Basic Linear Algebra Subprograms", "BLAS"),
    "CI/CD": ("continuous integration/continuous deployment", "CI/CD"),
    "CNN": ("convolutional neural network", "CNN"),
    "CTM": ("continuous therapeutic monitoring", "CTM"),
    "DAG": ("directed acyclic graph", "DAG"),
    "DCE": ("dead-code elimination", "DCE"),
    "ELT": ("extract, load, transform", "ELT"),
    "ETL": ("extract, transform, load", "ETL"),
    "FFT": ("fast Fourier transform", "FFT"),
    "GDPR": ("General Data Protection Regulation", "GDPR"),
    "GELU": ("Gaussian Error Linear Unit", "GELU"),
    "GEMM": ("general matrix multiply", "GEMM"),
    "HIPAA": ("Health Insurance Portability and Accountability Act", "HIPAA"),
    "HOG": ("histogram of oriented gradients", "HOG"),
    "i.i.d.": ("independent and identically distributed", "i.i.d."),
    "ICR": ("information-compute ratio", "ICR"),
    "ILSVRC": ("ImageNet Large Scale Visual Recognition Challenge", "ILSVRC"),
    "IOPS": ("input/output operations per second", "IOPS"),
    "IR": ("intermediate representation", "IR"),
    "JIT": ("just-in-time", "JIT"),
    "JSON": ("JavaScript Object Notation", "JSON"),
    "KWS": ("keyword spotting", "KWS"),
    "LLMs": ("large language models", "LLMs"),
    "LLM": ("large language model", "LLM"),
    "MAC": ("multiply-accumulate", "MAC"),
    "MIPS": ("microprocessor without interlocked pipelined stages", "MIPS"),
    "MLPs": ("multilayer perceptrons", "MLPs"),
    "MoE": ("mixture-of-experts", "MoE"),
    "NAS": ("neural architecture search", "NAS"),
    "NaN": ("not a number", "NaN"),
    "NVMe": ("Non-Volatile Memory Express", "NVMe"),
    "ONNX": ("Open Neural Network Exchange", "ONNX"),
    "OTA": ("over-the-air", "OTA"),
    "PTX": ("Parallel Thread Execution", "PTX"),
    "RBAC": ("role-based access control", "RBAC"),
    "ReLU": ("rectified linear unit", "ReLU"),
    "RISC": ("reduced instruction set computer", "RISC"),
    "RNN": ("recurrent neural network", "RNN"),
    "RNNs": ("recurrent neural networks", "RNNs"),
    "ROC": ("receiver operating characteristic", "ROC"),
    "SIFT": ("scale-invariant feature transform", "SIFT"),
    "SIMD": ("single instruction, multiple data", "SIMD"),
    "SLA": ("service level agreement", "SLA"),
    "SoC": ("system on chip", "SoC"),
    "SSA": ("static single-assignment", "SSA"),
    "TCO": ("total cost of ownership", "TCO"),
    "TFDV": ("TensorFlow Data Validation", "TFDV"),
    "TPU": ("Tensor Processing Unit", "TPU"),
    "UAT": ("universal approximation theorem", "UAT"),
    "ViT": ("vision transformer", "ViT"),
    "CNNs": ("convolutional neural networks", "CNNs"),
    "Adam": ("Adaptive Moment Estimation", "Adam"),
}

# Abbreviations that DON'T need expansion (well-known per style sheet)
NO_EXPAND = {"CUDA", "cuDNN", "GPU", "GPUs", "CPU", "CPUs", "RAM", "API",
             "APIs", "ML", "AI", "DNN", "DNNs", "USB", "IoT", "SDK",
             "HTTP", "HTTPS", "URL", "URLs", "PDF", "RGB", "LED",
             "PCIe", "DMA", "DRAM", "SRAM", "HBM", "FLOPS", "TFLOPS",
             "FP32", "FP16", "BF16", "INT8", "INT4"}

# Chapter QMD files (body chapters only, not frontmatter/backmatter)
CHAPTER_DIRS = [
    "introduction", "ml_systems", "ml_workflow", "data_engineering",
    "nn_computation", "nn_architectures", "frameworks", "training",
    "data_selection", "optimizations", "hw_acceleration", "benchmarking",
    "model_serving", "ml_ops", "responsible_engr", "conclusion",
]


def is_body_prose_line(line: str, in_code_fence: bool, in_yaml: bool) -> bool:
    """Return True if this line is body prose (not code/yaml/table/etc)."""
    if in_code_fence or in_yaml:
        return False
    stripped = line.lstrip()
    if stripped.startswith("#|"):
        return False
    if stripped.startswith("```"):
        return False
    if stripped.startswith(":::"):
        return False
    return True


def check_abbreviation_in_file(filepath: Path, abbr: str, expansion: str) -> dict:
    """Check if abbreviation is used in this file and if first use is expanded.

    Returns dict with keys: found, expanded, first_line, first_line_num
    """
    text = filepath.read_text(encoding="utf-8")
    lines = text.split("\n")

    in_code_fence = False
    in_yaml = False
    yaml_seen = 0

    # Build the expanded pattern to look for
    expanded_pattern = f"{expansion} ({abbr})"

    first_occurrence = None
    first_line_num = None
    is_expanded = False

    for idx, line in enumerate(lines, 1):
        stripped = line.strip()

        if stripped == "---":
            if yaml_seen == 0:
                in_yaml = True
                yaml_seen = 1
            elif yaml_seen == 1 and in_yaml:
                in_yaml = False
                yaml_seen = 2

        if stripped.startswith("```"):
            in_code_fence = not in_code_fence

        if not is_body_prose_line(line, in_code_fence, in_yaml):
            continue

        # Check for the abbreviation (as whole word)
        # Use word boundary for most, but handle special cases
        if abbr == "i.i.d.":
            pattern = r"i\.i\.d\."
        elif abbr == "IR":
            # IR is too common as substring — require word boundaries
            pattern = r"\bIR\b"
        else:
            pattern = r"\b" + re.escape(abbr) + r"\b"

        if re.search(pattern, line):
            if first_occurrence is None:
                first_occurrence = line.strip()
                first_line_num = idx
                # Check if this first use includes the expansion
                if expansion.lower() in line.lower():
                    is_expanded = True

    return {
        "found": first_occurrence is not None,
        "expanded": is_expanded,
        "first_line": first_occurrence,
        "first_line_num": first_line_num,
    }


def check_file(filepath: Path) -> list[dict]:
    """Check all abbreviations in a single file. Returns list of issues."""
    issues = []
    for abbr, (expansion, _) in ABBREVIATIONS.items():
        if abbr in NO_EXPAND:
            continue
        result = check_abbreviation_in_file(filepath, abbr, expansion)
        if result["found"] and not result["expanded"]:
            issues.append({
                "abbr": abbr,
                "expansion": expansion,
                "line_num": result["first_line_num"],
                "line": result["first_line"],
            })
    return issues


def main():
    parser = argparse.ArgumentParser(description="Check/fix abbreviation first-use expansion")
    parser.add_argument("path", type=Path, help="Directory to process")
    parser.add_argument("--check", action="store_true", help="Check only, don't fix")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be fixed")
    args = parser.parse_args()

    # Find chapter QMD files
    files = sorted(args.path.rglob("*.qmd"))

    total_issues = 0
    for f in files:
        issues = check_file(f)
        if issues:
            rel = f.relative_to(args.path) if args.path in f.parents else f
            for issue in issues:
                print(f"  {rel}:{issue['line_num']}: {issue['abbr']} — needs expansion to '{issue['expansion']} ({issue['abbr']})'")
                total_issues += 1

    print(f"\nFound {total_issues} abbreviations needing first-use expansion")


if __name__ == "__main__":
    main()

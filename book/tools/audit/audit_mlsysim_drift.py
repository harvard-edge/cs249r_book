#!/usr/bin/env python3
"""
audit_mlsysim_drift.py — find hand-coded values in chapter QMDs that should
come from mlsysim's canonical constants.

Approach:
  1. Build a registry of (canonical_name, numeric_value, magnitude_tolerance)
     from mlsysim.core.constants.
  2. Walk every chapter QMD's Python cells. For each numeric literal
     assignment, check if (a) the variable name resembles a canonical name
     fragment AND (b) the literal matches the canonical value (within
     tolerance to allow for unit conversions and rounding).
  3. Report drift.

This is a one-time audit (a recurring lint can be added separately to
audit_math_canonical if drift is significant).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Hand-curated registry of canonical (name fragment, value, tolerance, mlsysim path).
# Tolerance is fractional (0.02 = 2%) to allow for rounding and unit conversions.
# Names are *fragments* — we substring-match against variable identifiers in QMDs.
CANONICAL = [
    # GPU compute (TFLOP/s and TOPS) — registry paths for drift hints
    ("v100_fp16",      125,   0.02, "Hardware.Cloud.V100.compute.peak_flops"),
    ("v100_tflops",    125,   0.02, "Hardware.Cloud.V100.compute.peak_flops"),
    ("v100_fp32",      15.7,  0.05, 'Hardware.Cloud.V100.compute.precision_flops["fp32"]'),
    ("a100_fp16",      312,   0.02, "Hardware.Cloud.A100.compute.peak_flops"),
    ("a100_tflops",    312,   0.02, "Hardware.Cloud.A100.compute.peak_flops"),
    ("a100_tf32",      156,   0.02, 'Hardware.Cloud.A100.compute.precision_flops["tf32"]'),
    ("a100_fp32",      19.5,  0.05, 'Hardware.Cloud.A100.compute.precision_flops["fp32"]'),
    ("a100_int8",      624,   0.02, 'Hardware.Cloud.A100.compute.precision_flops["int8"]'),
    ("a100_sparse",    624,   0.02, 'Hardware.Cloud.A100.compute.precision_flops["fp16_sparse"]'),
    ("h100_fp16",      989,   0.02, "Hardware.Cloud.H100.compute.peak_flops"),
    ("h100_tflops",    989,   0.02, "Hardware.Cloud.H100.compute.peak_flops"),
    ("h100_fp8",       1979,  0.02, 'Hardware.Cloud.H100.compute.precision_flops["fp8"]'),
    ("h100_tf32",      494,   0.02, 'Hardware.Cloud.H100.compute.precision_flops["tf32"]'),
    ("h100_int8",      1979,  0.02, 'Hardware.Cloud.H100.compute.precision_flops["int8"]'),
    ("b200_fp16",      4500,  0.02, "Hardware.Cloud.B200.compute.peak_flops"),
    ("b200_fp8",       9000,  0.02, 'Hardware.Cloud.B200.compute.precision_flops["fp8"]'),
    ("mi300x_fp16",    1307,  0.02, "Hardware.Cloud.MI300X.compute.peak_flops"),

    # GPU memory (GiB, GB/s, TB/s)
    ("v100_mem_gb",    32,    0.05, "Hardware.Cloud.V100.memory.capacity"),
    ("v100_mem_bw",    900,   0.02, "Hardware.Cloud.V100.memory.bandwidth"),
    ("a100_mem_gb",    80,    0.05, "Hardware.Cloud.A100.memory.capacity"),
    ("a100_mem_bw",    2039,  0.02, "Hardware.Cloud.A100.memory.bandwidth"),
    ("a100_mem_tbs",   2.04,  0.02, "Hardware.Cloud.A100.memory.bandwidth (TB/s)"),
    ("h100_mem_gb",    80,    0.05, "Hardware.Cloud.H100.memory.capacity"),
    ("h100_mem_bw",    3350,  0.02, "Hardware.Cloud.H100.memory.bandwidth (GB/s)"),
    ("h100_mem_tbs",   3.35,  0.02, "Hardware.Cloud.H100.memory.bandwidth (TB/s)"),

    # GPU on-chip memory hierarchy + microarchitecture (MB / KiB / count)
    ("l2_cache",       50,    0.05, "Hardware.Cloud.H100.memory.l2_cache (MB)"),
    ("h100_l2",        50,    0.05, "Hardware.Cloud.H100.memory.l2_cache (MB)"),
    ("register_total", 33,    0.05, "Hardware.Cloud.H100 register file total = sm_count * register_file_per_sm (MiB)"),
    ("register_file",  256,   0.02, "Hardware.Cloud.H100.memory.register_file_per_sm (KiB)"),
    ("reg_per_sm",     256,   0.02, "Hardware.Cloud.H100.memory.register_file_per_sm (KiB)"),
    ("shared_mem",     228,   0.02, "Hardware.Cloud.H100.memory.shared_memory_per_sm (KiB)"),
    ("shared_per_sm",  228,   0.02, "Hardware.Cloud.H100.memory.shared_memory_per_sm (KiB)"),
    ("sm_count",       132,   0.02, "Hardware.Cloud.H100.compute.sm_count"),

    # On-chip / off-chip latency + access energy (ns / pJ) — Hardware.Tech / core/constants.py
    ("hbm_latency",    300,   0.05, "Hardware.Tech.Memory.HBM3.latency (ns)"),
    ("hbm_energy",     640,   0.05, "Hardware.Tech.Memory.HBM3.energy_per_access (pJ)"),
    ("hbm_access",     640,   0.05, "Hardware.Tech.Memory.HBM3.energy_per_access (pJ)"),
    ("h200_mem_gb",    141,   0.05, "Hardware.Cloud.H200.memory.capacity"),
    ("h200_mem_tbs",   4.8,   0.02, "Hardware.Cloud.H200.memory.bandwidth (TB/s)"),
    ("b200_mem_gb",    192,   0.05, "Hardware.Cloud.B200.memory.capacity"),
    ("b200_mem_tbs",   8,     0.05, "Hardware.Cloud.B200.memory.bandwidth (TB/s)"),

    # GPU power (W)
    ("v100_tdp",       300,   0.05, "Hardware.Cloud.V100.tdp"),
    ("a100_tdp",       400,   0.05, "Hardware.Cloud.A100.tdp"),
    ("h100_tdp",       700,   0.05, "Hardware.Cloud.H100.tdp"),
    ("h200_tdp",       700,   0.05, "Hardware.Cloud.H200.tdp"),
    ("b200_tdp",       1000,  0.05, "Hardware.Cloud.B200.tdp"),

    # Model parameters (count)
    ("gpt3_params_b",  175,   0.02, "Models.Language.GPT3.parameters (in B)"),
    ("gpt3_params_billion", 175, 0.02, "Models.Language.GPT3.parameters (in B)"),
    ("llama2_70b_params_b", 70, 0.02, "Models.Language.Llama2_70B.parameters (in B)"),
    ("llama2_70b_layers",   80, 0.05, "Models.Language.Llama2_70B.layers"),
    ("llama2_70b_hidden",   8192, 0.02, "Models.Language.Llama2_70B.hidden_dim"),
    ("llama2_70b_heads",    64, 0.05, "Models.Language.Llama2_70B.heads"),
    ("llama2_70b_kv_heads", 8,  0.05, "Models.Language.Llama2_70B.kv_heads"),
    ("gpt2_params_m",       1500, 0.05, "Models.GPT2.parameters (in M)"),

    # GPT-3 training (energy + ops)
    ("gpt3_train_energy_mwh", 1287, 0.05, "Models.Language.GPT3.training_energy_mwh"),
    ("gpt3_train_ops",       3.14e23, 0.05, "Models.Language.GPT3.training_ops"),

    # Interconnect bandwidth (GB/s)
    ("pcie_gen4_gb",   32,    0.05, "Hardware.Cloud.A100.interconnect.bandwidth (x16)"),
    ("pcie_gen5_gb",   64,    0.05, "Hardware.Cloud.H100.interconnect.bandwidth (x16)"),
    ("nvlink_h100",    900,   0.02, "Hardware.Cloud.H100.nvlink.bandwidth"),
    ("nvlink_a100",    600,   0.02, "Hardware.Cloud.A100.nvlink.bandwidth"),
]

# Python LEGO cells are fenced ```{python} ... ```
CELL_OPEN_RE = re.compile(r"^```\s*\{python\b[^}]*\}\s*$")
CELL_CLOSE_RE = re.compile(r"^```\s*$")

# Match `var = LITERAL` where LITERAL is a number (possibly with _N or scientific)
ASSIGN_RE = re.compile(
    r"^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*([0-9][0-9_]*(?:\.[0-9_]+)?(?:[eE][+\-]?[0-9]+)?)\b"
)

def python_cells(qmd_text: str):
    """Yield (start_line, line) for each line inside a {python} fence."""
    in_cell = False
    for i, line in enumerate(qmd_text.splitlines(), start=1):
        if not in_cell and CELL_OPEN_RE.match(line):
            in_cell = True
            continue
        if in_cell and CELL_CLOSE_RE.match(line):
            in_cell = False
            continue
        if in_cell:
            yield i, line

def parse_literal(s: str) -> float | None:
    try:
        return float(s.replace("_", ""))
    except ValueError:
        return None

def matches_canonical(var_name: str, value: float):
    """Return list of canonical matches for this (var, value) pair."""
    matches = []
    lower = var_name.lower()
    for fragment, canon_value, tol, path in CANONICAL:
        if fragment not in lower:
            continue
        # Magnitude tolerance check
        if canon_value == 0:
            continue
        rel_error = abs(value - canon_value) / canon_value
        if rel_error <= tol:
            matches.append((fragment, canon_value, path))
    return matches

def audit_file(path: Path) -> list:
    text = path.read_text(encoding="utf-8", errors="ignore")
    findings = []
    for lineno, line in python_cells(text):
        # Skip comments (a bare assignment in a comment shouldn't fire)
        if line.lstrip().startswith("#"):
            continue
        m = ASSIGN_RE.match(line)
        if not m:
            continue
        var, lit = m.group(1), m.group(2)
        val = parse_literal(lit)
        if val is None:
            continue
        canon = matches_canonical(var, val)
        if canon:
            findings.append((path, lineno, line.rstrip(), var, val, canon))
    return findings

def main():
    root = Path("book/quarto/contents")
    if not root.exists():
        print(f"ERROR: run from MLSysBook root (expected {root})", file=sys.stderr)
        sys.exit(2)
    qmds = sorted(root.rglob("*.qmd"))
    all_findings = []
    for q in qmds:
        all_findings.extend(audit_file(q))
    if not all_findings:
        print("✓ no hand-coded canonical-constant drift detected")
        return 0
    # Group by file
    print(f"Found {len(all_findings)} potential drift sites:\n")
    last_file = None
    for path, lineno, line, var, val, canon in all_findings:
        if path != last_file:
            print(f"\n📄 {path}")
            last_file = path
        canon_str = ", ".join(f"{p}" for _, _, p in canon)
        print(f"  L{lineno}: {var} = {val}")
        print(f"           → matches mlsysim.{canon_str}")
        print(f"           {line.strip()}")
    print(f"\nTotal: {len(all_findings)} drift candidates")
    return 1

if __name__ == "__main__":
    sys.exit(main())

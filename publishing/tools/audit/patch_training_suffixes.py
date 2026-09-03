#!/usr/bin/env python3
"""Add missing fmt suffixes in training.qmd (idempotent)."""
import re
from pathlib import Path

QMD = Path(__file__).resolve().parents[3] / "book/quarto/contents/vol1/training/training.qmd"

RULES = [
    (r"(\w+_gb_str)\s*=\s*fmt\(([^)]*)\)", " GB"),
    (r"(\w+_mb_str)\s*=\s*fmt\(([^)]*)\)", " MB"),
    (r"(\w+_tflops_str)\s*=\s*fmt\(([^)]*)\)", " TFLOP/s"),
    (r"(\w+_pflops_str)\s*=\s*fmt\(([^)]*)\)", " PFLOP/s"),
    (r"(\w+_gbs_str)\s*=\s*fmt\(([^)]*)\)", " GB/s"),
    (r"(\w+_tbs_str)\s*=\s*fmt\(([^)]*)\)", " TB/s"),
    (r"(\w+_tb_s_str)\s*=\s*fmt\(([^)]*)\)", " TB/s"),
    (r"(\w+_pct_str)\s*=\s*fmt\(([^)]*)\)", "%"),
    (r"(\w+_gpus_str)\s*=\s*fmt\(([^)]*)\)", " GPUs"),
    (r"(\w+_gpu_str)\s*=\s*fmt\(([^)]*)\)", " GPUs"),
    (r"(\w+_hours_str)\s*=\s*fmt\(([^)]*)\)", " hours"),
    (r"(\w+_days_str)\s*=\s*fmt\(([^)]*)\)", " days"),
    (r"(\w+_kwh_str)\s*=\s*fmt\(([^)]*)\)", " kWh"),
    (r"(\w+_w_str)\s*=\s*fmt\(([^)]*)\)", " W"),
    (r"(\w+_tdp_str)\s*=\s*fmt\(([^)]*)\)", " W"),
    (r"(\w+_ms_str)\s*=\s*fmt\(([^)]*)\)", " ms"),
    (r"(\w+_us_str)\s*=\s*fmt\(([^)]*)\)", " μs"),
    (r"(\w+_s_str)\s*=\s*fmt\(([^)]*)\)", " s"),
    (r"(\w+_params_b_str)\s*=\s*fmt\(([^)]*)\)", " B"),
    (r"(\w+_speedup_str)\s*=\s*fmt\(([^)]*)\)", "×"),
    (r"(\w+_reduction_str)\s*=\s*fmt\(([^)]*)\)", "×"),
    (r"(\w+_factor_str)\s*=\s*fmt\(([^)]*)\)", "×"),
    (r"(\w+_multiplier_str)\s*=\s*fmt\(([^)]*)\)", "×"),
    (r"(\w+_util_str)\s*=\s*fmt\(([^)]*)\)", "%"),
    (r"(\w+_overhead_str)\s*=\s*fmt\(([^)]*)\)", "%"),
]

def patch(text: str) -> tuple[str, int]:
    n = 0
    for pat, suf in RULES:
        def repl(m: re.Match[str]) -> str:
            nonlocal n
            if "suffix=" in m.group(0):
                return m.group(0)
            n += 1
            inner = m.group(2).rstrip()
            return f"{m.group(1)} = fmt({inner}, suffix={suf!r})"
        text = re.sub(pat + r"(?!.*suffix=)", repl, text)
    return text, n

def main():
    text = QMD.read_text()
    text, n = patch(text)
    text = text.replace("suffix=' B parameters'", "suffix=' B'")
    QMD.write_text(text)
    print(f"Patched {n} lines")

if __name__ == "__main__":
    main()

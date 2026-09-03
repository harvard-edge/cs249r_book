#!/usr/bin/env python3
"""Add fmt suffixes to training.qmd LEGO exports from naming conventions."""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
QMD = REPO / "book/quarto/contents/vol1/training/training.qmd"

SUFFIX_RULES: list[tuple[str, str]] = [
    (r"_gbs_str$", " GB/s"),
    (r"_tbs_str$", " TB/s"),
    (r"_tflops_str$", " TFLOP/s"),
    (r"_pflops_str$", " PFLOP/s"),
    (r"_flops_eqn_str$", ""),
    (r"_eqn_str$", ""),
    (r"_formula_str$", ""),
    (r"_math_str$", ""),
    (r"_sci_str$", ""),
    (r"_tb_s_str$", " TB/s"),
    (r"_time_str$", " s"),
    (r"_time_s_str$", " s"),
    (r"_time_ms_str$", " ms"),
    (r"_time_us_str$", " μs"),
    (r"_exec_time_str$", " μs"),
    (r"_ms_str$", " ms"),
    (r"_us_str$", " μs"),
    (r"_hours_str$", " hours"),
    (r"_hrs_str$", " hours"),
    (r"_days_str$", " days"),
    (r"_weeks_str$", " weeks"),
    (r"_months_str$", " months"),
    (r"_years_str$", " years"),
    (r"_kwh_str$", " kWh"),
    (r"_tdp_str$", " W"),
    (r"(?<![a-z])_w_str$", " W"),
    (r"_qps_str$", " QPS"),
    (r"_sps_str$", " samples/s"),
    (r"_rps_str$", " req/s"),
    (r"_tokens_str$", " tokens"),
    (r"_params_b_str$", " B"),
    (r"_params_str$", " parameters"),
    (r"_b_str$", " B"),
    (r"_layers_str$", " layers"),
    (r"_heads_str$", " heads"),
    (r"_steps_str$", " steps"),
    (r"_images_str$", " images"),
    (r"_speedup_str$", "×"),
    (r"_reduction_str$", "×"),
    (r"_multiplier_str$", "×"),
    (r"_ratio_str$", "×"),
    (r"_over_v100_str$", "×"),
    (r"_overhead_str$", "%"),
    (r"_util_str$", "%"),
    (r"_mfu_str$", ""),
    (r"_ridge_str$", " FLOP/byte"),
    (r"_intensity_str$", " FLOP/byte"),
    (r"_flop_byte_str$", " FLOP/byte"),
    (r"_co2_str$", " tons CO₂"),
    (r"_energy_str$", " kWh"),
    (r"_energy_saved_str$", " kWh"),
    (r"_household_months_str$", " months"),
    (r"_hourly_str$", "/hour"),
    (r"_2wk_str$", " K"),
    (r"_cost_str$", ""),
    (r"_usd_str$", ""),
    (r"_gb_str$", " GB"),
    (r"_mb_str$", " MB"),
    (r"_kb_str$", " KB"),
    (r"_bw_str$", " GB/s"),
    (r"(?<![a-z])_gpus_str$", " GPUs"),
    (r"(?<![a-z])_gpu_str$", " GPUs"),
    (r"_hosts_str$", " hosts"),
    (r"_pct_str$", "%"),
    (r"_percent_str$", "%"),
    (r"(?<![a-z])_s_str$", " s"),
]

SKIP_ATTRS = {
    "attn_bytes_formula_str",
    "local_grad_eqn_str",
    "global_grad_eqn_str",
    "cf_flops_eqn_str",
    "gpt2_fwd_flops_str",
    "gpt2_total_flops_str",
    "mnist_total_flops_str",
    "mnist_mem_traffic_str",
    "mac_factor_str",
    "allreduce_factor_str",
    "batch_per_gpu_str",
    "micro_batch_str",
    "effective_batch_str",
    "physical_batch_str",
    "ffn_coeff_str",
    "n_projections_str",
}

EXPLICIT_SUFFIX: dict[str, str] = {
    "gradient_size_str": " GB",
    "allreduce_str": " GB",
    "network_bw_str": " GB/s",
    "nvlink_h100_str": " GB/s",
    "pcie_gen3_str": " GB/s",
    "random_access_gbs_str": " GB/s",
    "nvme_bw_str": " GB/s",
    "token_rate_str": " tokens/s",
    "tokens_per_batch_str": " tokens",
    "fp_model_20b_params_str": " B",
    "fp_model_7b_params_str": " B",
    "fp_data_threshold_m_str": " million",
    "fp_gpu_count_str": " GPUs",
    "fp_prefetch_reduction_str": "%",
    "fp_overfit_degrade_min_str": "%",
    "fp_overfit_degrade_max_str": "%",
    "fp_util_batch_256_str": "%",
    "fp_util_diff_str": "%",
    "fp_memory_reduction_str": "×",
    "fp_mp_speedup_v100_str": "×",
}

FMT_LINE = re.compile(
    r"^(\s*)(\w+_str)\s*=\s*(fmt(?:_int|_percent)?)\((.+)\)\s*(#.*)?$"
)


def infer_suffix(attr: str) -> str | None:
    if attr in SKIP_ATTRS:
        return None
    if attr in EXPLICIT_SUFFIX:
        return EXPLICIT_SUFFIX[attr]
    if "_gb" in attr and attr.endswith("_str"):
        return " GB"
    if "_mb" in attr and attr.endswith("_str"):
        return " MB"
    if "_kb" in attr and attr.endswith("_str"):
        return " KB"
    if "fp_model" in attr and attr.endswith("_params_str"):
        return " B"
    for pat, suf in SUFFIX_RULES:
        if re.search(pat, attr):
            return suf if suf else None
    if "util" in attr and attr.endswith("_str"):
        return "%"
    if "pct" in attr or "percent" in attr:
        return "%"
    return None


def add_suffix_to_call(fn: str, inner: str, suffix: str) -> str:
    if "suffix=" in inner:
        return inner
    inner = inner.rstrip()
    if fn == "fmt_percent":
        return inner
    return f"{inner}, suffix={suffix!r}"


def migrate(text: str) -> tuple[str, int]:
    lines = text.splitlines(keepends=True)
    changed = 0
    out: list[str] = []
    for line in lines:
        m = FMT_LINE.match(line.rstrip("\n"))
        if not m:
            out.append(line)
            continue
        indent, attr, fn, inner, comment = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5) or ""
        suf = infer_suffix(attr)
        if suf is None or "suffix=" in inner:
            out.append(line)
            continue
        new_inner = add_suffix_to_call(fn, inner, suf)
        nl = f"{indent}{attr} = {fn}({new_inner}){comment}\n"
        if nl != line:
            changed += 1
        out.append(nl)
    return "".join(out), changed


def main() -> int:
    text = QMD.read_text(encoding="utf-8")
    new_text, n = migrate(text)
    if n:
        QMD.write_text(new_text, encoding="utf-8")
    print(f"Added suffix to {n} fmt assignments in {QMD.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

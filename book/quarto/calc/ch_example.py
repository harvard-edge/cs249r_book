"""
Chapter Calculation Module — Example Pattern
=============================================
Each chapter gets a file like this. It imports constants, computes every
derived number the chapter needs, and exposes them as a flat namespace.

Usage in .qmd:
    ```{python}
    #| echo: false
    from ch_example import C   # one import, all numbers
    ```
    Then inline: `{python} C.bw_a100_vs_v100`
"""

from constants import *


class C:
    """
    All computed values for this chapter, organized by section.

    Naming convention:
        section__variable_name
    The double underscore groups by section for readability,
    while keeping them all accessible as C.section__name.

    For simple cases, skip the section prefix.
    """

    # ── Section: Hardware Specs (direct from constants, formatted) ────
    v100_tflops = f"{V100_FLOPS_FP16_TENSOR / TB:.0f}"       # "125"
    a100_tflops = f"{A100_FLOPS_FP16_TENSOR / TB:.0f}"       # "312"
    h100_tflops = f"{H100_FLOPS_FP16_TENSOR / TB:.0f}"       # "989"

    v100_bw_gbs = f"{V100_MEM_BW / GB:.0f}"                  # "900"
    a100_bw_gbs = f"{A100_MEM_BW / GB:.0f}"                  # "2039"
    h100_bw_tbs = f"{H100_MEM_BW / TB:.2f}"                  # "3.35"

    v100_mem_gib = f"{V100_MEM_CAPACITY / GiB:.0f}"           # "32"
    a100_mem_gib = f"{A100_MEM_CAPACITY / GiB:.0f}"           # "80"
    h100_mem_gib = f"{H100_MEM_CAPACITY / GiB:.0f}"           # "80"

    # ── Section: Bandwidth Ratios ─────────────────────────────────────
    _bw_a100_v100 = A100_MEM_BW / V100_MEM_BW
    _bw_h100_v100 = H100_MEM_BW / V100_MEM_BW
    _bw_h100_a100 = H100_MEM_BW / A100_MEM_BW

    bw_a100_vs_v100 = f"{_bw_a100_v100:.1f}"                 # "2.3"
    bw_h100_vs_v100 = f"{_bw_h100_v100:.1f}"                 # "3.7"
    bw_h100_vs_a100 = f"{_bw_h100_a100:.1f}"                 # "1.6"

    # ── Section: Compute Ratios ───────────────────────────────────────
    _comp_a100_v100 = A100_FLOPS_FP16_TENSOR / V100_FLOPS_FP16_TENSOR
    _comp_h100_v100 = H100_FLOPS_FP16_TENSOR / V100_FLOPS_FP16_TENSOR
    _comp_h100_a100 = H100_FLOPS_FP16_TENSOR / A100_FLOPS_FP16_TENSOR

    compute_a100_vs_v100 = f"{_comp_a100_v100:.1f}"           # "2.5"
    compute_h100_vs_v100 = f"{_comp_h100_v100:.1f}"           # "7.9"
    compute_h100_vs_a100 = f"{_comp_h100_a100:.1f}"           # "3.2"

    # Memory wall: compute grew faster than bandwidth
    memory_wall_gap = f"{_comp_h100_v100 / _bw_h100_v100:.1f}"  # "2.1"

    # ── Section: Ridge Points (Roofline Model) ────────────────────────
    _ridge_v100 = V100_FLOPS_FP16_TENSOR / V100_MEM_BW
    _ridge_a100 = A100_FLOPS_FP16_TENSOR / A100_MEM_BW
    _ridge_h100 = H100_FLOPS_FP16_TENSOR / H100_MEM_BW

    ridge_v100 = f"{_ridge_v100:.0f}"                         # "139"
    ridge_a100 = f"{_ridge_a100:.0f}"                         # "153"
    ridge_h100 = f"{_ridge_h100:.0f}"                         # "295"

    # ── Section: Energy ───────────────────────────────────────────────
    energy_ratio = f"{ENERGY_DRAM_ACCESS_PJ / ENERGY_FLOP_PJ:.0f}"  # "139"
    energy_dram_pj = f"{ENERGY_DRAM_ACCESS_PJ}"               # "640"
    energy_flop_pj = f"{ENERGY_FLOP_PJ}"                      # "4.6"

    # ── Section: ResNet-50 ────────────────────────────────────────────
    resnet_gflops = f"{RESNET50_FLOPs / 1e9:.1f}"             # "4.1"
    resnet_params_m = f"{RESNET50_PARAMS / 1e6:.1f}"           # "25.6"

    _resnet_time_v100 = RESNET50_FLOPs / V100_FLOPS_FP32      # seconds
    _resnet_time_a100 = RESNET50_FLOPs / A100_FLOPS_TF32      # seconds
    resnet_latency_v100_us = f"{_resnet_time_v100 / US:.0f}"   # "261"
    resnet_latency_a100_us = f"{_resnet_time_a100 / US:.0f}"   # "26"
    resnet_speedup_a100_v100 = f"{_resnet_time_v100 / _resnet_time_a100:.1f}"  # "9.9"

    # ── Section: Llama-3-8B Memory ────────────────────────────────────
    _llama_fp16 = LLAMA3_8B_PARAMS * 2
    _llama_int8 = LLAMA3_8B_PARAMS * 1
    _llama_int4 = LLAMA3_8B_PARAMS * 0.5

    llama_params_b = f"{LLAMA3_8B_PARAMS / 1e9:.0f}"           # "8"
    llama_fp16_gib = f"{_llama_fp16 / GiB:.1f}"                # "14.9"
    llama_int8_gib = f"{_llama_int8 / GiB:.1f}"                # "7.5"
    llama_int4_gib = f"{_llama_int4 / GiB:.1f}"                # "3.7"
    llama_quant_reduction = f"{_llama_fp16 / _llama_int4:.0f}"  # "4"

    llama_fits_v100_fp16 = "✓" if _llama_fp16 < V100_MEM_CAPACITY else "✗"
    llama_fits_t4_fp16 = "✓" if _llama_fp16 < 16 * GiB else "✗"
    llama_fits_t4_int8 = "✓" if _llama_int8 < 16 * GiB else "✗"

    # ── Section: Interconnects ────────────────────────────────────────
    _transfer_time = lambda bytes, bw: bytes / bw  # seconds

    pcie3_bw_gbs = f"{PCIE_GEN3_BW / GB:.0f}"
    pcie5_bw_gbs = f"{PCIE_GEN5_BW / GB:.0f}"
    nvlink_bw_gbs = f"{NVLINK_BW / GB:.0f}"
    net100g_bw_gbs = f"{NETWORK_100G_BW / GB:.1f}"

    # Transfer Llama-3-8B FP16
    llama_transfer_pcie3_ms = f"{_llama_fp16 / PCIE_GEN3_BW / MS:.0f}"
    llama_transfer_pcie5_ms = f"{_llama_fp16 / PCIE_GEN5_BW / MS:.0f}"
    llama_transfer_nvlink_ms = f"{_llama_fp16 / NVLINK_BW / MS:.0f}"
    llama_transfer_net100g_ms = f"{_llama_fp16 / NETWORK_100G_BW / MS:.0f}"

    nvlink_vs_pcie4 = f"{NVLINK_BW / PCIE_GEN4_BW:.0f}"        # "19"

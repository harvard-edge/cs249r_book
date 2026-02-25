import math
from mlsys.constants import (
    H100_MEM_BW, H100_FLOPS_FP16_TENSOR, NVME_SEQUENTIAL_BW,
    GPT3_PARAMS, BILLION, MILLION, TRILLION, THOUSAND,
    GB, TB, second, byte, flop, USD, kilowatt, hour,
    GPUS_PER_HOST, SEC_PER_DAY, SEC_PER_YEAR,
    H100_MEM_CAPACITY, BYTES_FP16, BYTES_FP32,
    US, Mparam, RESNET50_PARAMS,
    NVLINK_H100_BW, PCIE_GEN5_BW,
    A100_MEM_CAPACITY, H100_FLOPS_FP8_TENSOR, H100_TDP
)
from mlsys.formatting import fmt, check, md

class StorageHierarchyAnalysis:
    """
    Namespace for global storage hierarchy and pipeline calculations.
    """
    # 1. LOAD
    h100_bw = H100_MEM_BW
    h100_flops_fp16 = H100_FLOPS_FP16_TENSOR
    h100_cap = H100_MEM_CAPACITY
    nvme_bw_raw = NVME_SEQUENTIAL_BW
    pfs_node_bw = (4.0 * GB / second)
    s3_bw = (1.0 * GB / second)

    gpt3_params = GPT3_PARAMS.m_as('param')
    t_step_ms = 200

    n_gpus_image = 256
    img_size = 150 * THOUSAND
    batch_img_gpu = 256
    util_target = 0.80

    dataset_size_tb = 100
    cost_s3_gb_mo = 0.02
    cost_nvme_gb_mo = 0.10
    cost_glacier_gb_mo = 0.004
    cost_egress_gb = 0.09

    n_tail_servers = 100
    p_tail_fail = 0.01

    # 2. EXECUTE
    t_step_s = t_step_ms / 1000
    req_bw_imagenet_val = (n_gpus_image * util_target * (batch_img_gpu * img_size)) / t_step_s
    req_bw_imagenet_gbs = req_bw_imagenet_val / BILLION

    t_comp_val = 200
    t_io_val = 250
    stall_max_t = max(t_comp_val, t_io_val)
    data_stall_pct_val = ((stall_max_t - t_comp_val) / stall_max_t) * 100

    bytes_per_param_ckpt = BYTES_FP16.m_as(byte) + (2 * BYTES_FP32.m_as(byte))
    ckpt_total_gb_val = (gpt3_params * bytes_per_param_ckpt) / BILLION

    n_nodes = 256
    node_shard_gb = ckpt_total_gb_val / n_nodes
    ckpt_nvme_s_val = node_shard_gb / (4 * nvme_bw_raw.m_as(GB/second))
    ckpt_pfs_s_val = node_shard_gb / pfs_node_bw.m_as(GB/second)

    s3_annual_val = dataset_size_tb * 1000 * cost_s3_gb_mo * 12
    nvme_annual_val = dataset_size_tb * 1000 * cost_nvme_gb_mo * 12
    glacier_annual_val = dataset_size_tb * 1000 * cost_glacier_gb_mo * 12
    tier_cost_ratio_val = cost_nvme_gb_mo / cost_s3_gb_mo
    egress_100tb_val = dataset_size_tb * 1000 * cost_egress_gb

    prob_tail_all = (1.0 - p_tail_fail) ** n_tail_servers

    images_per_sec = 1000
    raw_bw_val = (images_per_sec * img_size) / MILLION
    hdd_iops = 100
    hdd_slowdown_val = images_per_sec / hdd_iops

    gds_trad_us = 120
    gds_bypass_us = 30

    # 3. GUARD
    check(req_bw_imagenet_gbs > 10, f"ImageNet aggregate BW should be high, got {req_bw_imagenet_gbs:.1f} GB/s")
    check(ckpt_total_gb_val > 1000, "175B checkpoint must be > 1 TB")
    check(data_stall_pct_val == 20, "Data stall calculation mismatch")

    # 4. OUTPUT
    h100_bw_tbs = f"{h100_bw.m_as(TB/second):.2f}"
    nvme_bw_str = f"{nvme_bw_raw.m_as(GB/second):.1f}"
    gpt3_params_b = f"{gpt3_params / BILLION:.0f}"
    h100_hbm_cap_gb = f"{h100_cap.m_as(GB):.0f}"
    req_bw_imagenet_str = f"{req_bw_imagenet_gbs:.1f}"
    data_stall_pct_str = f"{data_stall_pct_val:.0f}"
    ckpt_total_gb_str = f"{ckpt_total_gb_val:,.0f}"
    ckpt_nvme_s_str = f"{ckpt_nvme_s_val:.1f}"
    ckpt_pfs_s_str = f"{ckpt_pfs_s_val:.1f}"
    s3_annual_str = f"{s3_annual_val:,.0f}"
    nvme_annual_str = f"{nvme_annual_val:,.0f}"
    glacier_annual_str = f"{glacier_annual_val:,.0f}"
    tier_cost_ratio_str = f"{tier_cost_ratio_val:.0f}"
    egress_100tb_str = f"{egress_100tb_val:,.0f}"
    prob_tail_all_str = f"{prob_tail_all:.3f}"
    fail_rate_pct_str = f"{p_tail_fail * 100:.0f}"
    n_tail_servers_str = f"{n_tail_servers}"
    raw_bw_str = f"{raw_bw_val:.0f}"
    hdd_slowdown_factor = f"{hdd_slowdown_val:.0f}"
    gds_trad_us_str = f"{gds_trad_us}"
    gds_bypass_us_str = f"{gds_bypass_us}"
    t_comp_stall_str = f"{t_comp_val}"
    t_io_stall_str = f"{t_io_val}"

# EXPORTS
h100_bw_tbs = StorageHierarchyAnalysis.h100_bw_tbs
nvme_bw = StorageHierarchyAnalysis.nvme_bw_str
gpt3_params_b = StorageHierarchyAnalysis.gpt3_params_b
h100_hbm_cap_gb = StorageHierarchyAnalysis.h100_hbm_cap_gb
req_bw_imagenet = StorageHierarchyAnalysis.req_bw_imagenet_str
data_stall_pct_str = StorageHierarchyAnalysis.data_stall_pct_str
stall_pct_display_math = md(
    f"$$	ext{{Stall \%}} = \frac{{{StorageHierarchyAnalysis.t_io_val} - {StorageHierarchyAnalysis.t_comp_val}}}{{{StorageHierarchyAnalysis.t_io_val}}} = "
    f"\mathbf{{{data_stall_pct_str}\%}}$$"
)
ckpt_total_gb = StorageHierarchyAnalysis.ckpt_total_gb_str
ckpt_nvme_s = StorageHierarchyAnalysis.ckpt_nvme_s_str
ckpt_pfs_s = StorageHierarchyAnalysis.ckpt_pfs_s_str
s3_annual_cost = StorageHierarchyAnalysis.s3_annual_str
nvme_annual_cost = StorageHierarchyAnalysis.nvme_annual_str
glacier_annual_cost = StorageHierarchyAnalysis.glacier_annual_str
tier_cost_ratio = StorageHierarchyAnalysis.tier_cost_ratio_str
egress_100tb_cost = StorageHierarchyAnalysis.egress_100tb_str
prob_tail_all_str = StorageHierarchyAnalysis.prob_tail_all_str
fail_rate_pct_str = StorageHierarchyAnalysis.fail_rate_pct_str
n_tail_servers_str = StorageHierarchyAnalysis.n_tail_servers_str
raw_bw_str = StorageHierarchyAnalysis.raw_bw_str
hdd_slowdown_factor = StorageHierarchyAnalysis.hdd_slowdown_factor
gds_trad_us = StorageHierarchyAnalysis.gds_trad_us_str
gds_bypass_us = StorageHierarchyAnalysis.gds_bypass_us_str
t_comp_stall_str = StorageHierarchyAnalysis.t_comp_stall_str
t_io_stall_str = StorageHierarchyAnalysis.t_io_stall_str
a100_mem = f"{A100_MEM_CAPACITY.m_as(GB):.0f}"
h100_mem = f"{H100_MEM_CAPACITY.m_as(GB):.0f}"
h100_fp8_tflops = f"{H100_FLOPS_FP8_TENSOR.m_as(TFLOPs/second):,.0f}"
h100_fp16_tflops = f"{H100_FLOPS_FP16_TENSOR.m_as(TFLOPs/second):,.0f}"
h100_tdp_w = f"{H100_TDP.m_as(watt):.0f}"
resnet_params_m = f"{RESNET50_PARAMS.m_as(Mparam):.1f}"
nvlink_bw_gbs = f"{NVLINK_H100_BW.m_as(GB/second):.0f}"
pcie5_bw_gbs = f"{PCIE_GEN5_BW.m_as(GB/second):.0f}"

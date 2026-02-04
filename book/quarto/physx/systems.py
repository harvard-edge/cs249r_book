"""
systems.py
System profiles used across the book (single-node and edge tiers).
"""

from .constants import (
    A100_FLOPS_FP16_TENSOR,
    A100_MEM_BW,
    A100_MEM_CAPACITY,
    A100_TDP,
    H100_FLOPS_FP16_TENSOR,
    H100_MEM_BW,
    H100_MEM_CAPACITY,
    H100_TDP,
    MOBILE_NPU_TOPS_INT8,
    MOBILE_NPU_MEM_BW,
    MOBILE_TDP_W,
    KWS_DSCNN_PARAMS,
)


SYSTEMS = {
    "a100_single_node": {
        "label": "A100 Single-Node",
        "peak_flops": A100_FLOPS_FP16_TENSOR,
        "mem_bw": A100_MEM_BW,
        "mem_capacity": A100_MEM_CAPACITY,
        "tdp": A100_TDP,
    },
    "h100_single_node": {
        "label": "H100 Single-Node",
        "peak_flops": H100_FLOPS_FP16_TENSOR,
        "mem_bw": H100_MEM_BW,
        "mem_capacity": H100_MEM_CAPACITY,
        "tdp": H100_TDP,
    },
    "mobile_soc": {
        "label": "Mobile SoC",
        "peak_flops": MOBILE_NPU_TOPS_INT8,
        "mem_bw": MOBILE_NPU_MEM_BW,
        "tdp": MOBILE_TDP_W,
    },
    "tinyml_device": {
        "label": "TinyML Device",
        "notes": "Profile for always-on KWS-class workloads.",
        "params_budget": KWS_DSCNN_PARAMS,
    },
}

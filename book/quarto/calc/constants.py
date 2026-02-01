# constants.py
# The "Physics Engine" of Machine Learning Systems
# This file defines the single source of truth for hardware specifications,
# constants, and conversion factors used throughout the textbook.

import pint
ureg = pint.UnitRegistry()
Q_ = ureg.Quantity

# --- Units ---
byte = ureg.byte
second = ureg.second
joule = ureg.joule
watt = ureg.watt
meter = ureg.meter
hour = ureg.hour
day = ureg.day

# Register data-scale aliases so .to(TB), .to(GB/second), etc. work
ureg.define('KB = 1e3 * byte')
ureg.define('MB = 1e6 * byte')
ureg.define('GB = 1e9 * byte')
ureg.define('TB = 1e12 * byte')
ureg.define('PB = 1e15 * byte')

KB = ureg.KB
MB = ureg.MB
GB = ureg.GB
TB = ureg.TB
PB = ureg.PB

# Binary units (pint has kibibyte etc. built-in, register short aliases)
ureg.define('KiB = 1024 * byte')
ureg.define('MiB = 1048576 * byte')
ureg.define('GiB = 1073741824 * byte')
ureg.define('TiB = 1099511627776 * byte')

KiB = ureg.KiB
MiB = ureg.MiB
GiB = ureg.GiB
TiB = ureg.TiB

# --- Time (registered so .to(MS) scales magnitudes correctly) ---
ureg.define('MS = 1e-3 * second')
ureg.define('US = 1e-6 * second')
ureg.define('NS = 1e-9 * second')

MS = ureg.MS
US = ureg.US
NS = ureg.NS

# --- Hardware Specifications (The Silicon Contract) ---

# FLOPs are dimensionless "operations"
ureg.define('flop = 1 * count')
ureg.define('GFLOPs = 1e9 * flop')
ureg.define('TFLOPs = 1e12 * flop')
ureg.define('ZFLOPs = 1e21 * flop')

flop = ureg.flop
GFLOPs = ureg.GFLOPs
TFLOPs = ureg.TFLOPs
ZFLOPs = ureg.ZFLOPs

# NVIDIA V100 (Volta, 2017) — Source: NVIDIA V100 Data Sheet
V100_FLOPS_FP16_TENSOR = 125 * TFLOPs / second
V100_FLOPS_FP32 = 15.7 * TFLOPs / second
V100_MEM_BW = 900 * GB / second           # HBM2
V100_MEM_CAPACITY = 32 * GiB
V100_TDP = 300 * watt                     # SXM2 variant

# NVIDIA A100 (Ampere, 2020) — Source: NVIDIA A100 Data Sheet
A100_FLOPS_FP16_TENSOR = 312 * TFLOPs / second
A100_FLOPS_TF32 = 156 * TFLOPs / second
A100_FLOPS_FP32 = 19.5 * TFLOPs / second  # Standard CUDA cores
A100_FLOPS_INT8 = 624 * TFLOPs / second   # INT8 Tensor Core
A100_MEM_BW = 2039 * GB / second           # HBM2e (SXM variant)
A100_MEM_CAPACITY = 80 * GiB              # SXM variant (also 40 GiB PCIe)
A100_TDP = 400 * watt                     # SXM variant

# NVIDIA H100 (Hopper, 2022) — Source: NVIDIA H100 Data Sheet
H100_FLOPS_FP16_TENSOR = 989 * TFLOPs / second
H100_FLOPS_FP8_TENSOR = 1979 * TFLOPs / second
H100_FLOPS_TF32 = 756 * TFLOPs / second
H100_FLOPS_INT8 = 3958 * TFLOPs / second  # INT8 Tensor Core
H100_MEM_BW = 3.35 * TB / second          # HBM3
H100_MEM_CAPACITY = 80 * GiB
H100_TDP = 700 * watt                     # SXM variant

# NVIDIA B100/B200 (Blackwell, 2024) — Source: NVIDIA Blackwell Architecture
B200_FLOPS_FP16_TENSOR = 4500 * TFLOPs / second
B200_FLOPS_FP8_TENSOR = 9000 * TFLOPs / second
B200_MEM_BW = 8 * TB / second             # HBM3e
B200_MEM_CAPACITY = 192 * GiB
B200_TDP = 1000 * watt

# NVIDIA T4 (Turing, 2018) — Source: NVIDIA T4 Data Sheet
T4_FLOPS_FP16_TENSOR = 65 * TFLOPs / second
T4_FLOPS_INT8 = 130 * TFLOPs / second
T4_MEM_BW = 320 * GB / second
T4_TDP = 70 * watt

# Google TPU v4 — Source: Google TPUv4 paper (Jouppi et al., 2023)
TPUV4_FLOPS_BF16 = 275 * TFLOPs / second
TPUV4_MEM_BW = 1200 * GB / second

# High-end Desktop CPU (Reference)
CPU_FLOPS_FP32 = 1 * TFLOPs / second

# Mobile NPU
MOBILE_NPU_TOPS_INT8 = 35 * TFLOPs / second
MOBILE_NPU_MEM_BW = 100 * GB / second

# --- Network & Interconnect ---
ureg.define('Gbps = 1e9 * bit / second')
Gbps = ureg.Gbps
NETWORK_10G_BW = 10 * Gbps
NETWORK_100G_BW = 100 * Gbps
NETWORK_5G_ENERGY_PER_MB_MJ = 100 * ureg.millijoule / MB

# Intra-node interconnects
NVLINK_V100_BW = 300 * GB / second        # NVLink 2.0 (V100, 6 links × 50 GB/s)
NVLINK_A100_BW = 600 * GB / second        # NVLink 3.0 (A100, 12 links × 50 GB/s)
NVLINK_H100_BW = 900 * GB / second        # NVLink 4.0 (H100, 18 links × 50 GB/s)
PCIE_GEN4_BW = 32 * GB / second           # PCIe Gen4 x16 (bidirectional)
PCIE_GEN5_BW = 64 * GB / second           # PCIe Gen5 x16 (bidirectional)

# Inter-node interconnects
INFINIBAND_HDR_BW = 200 * Gbps            # HDR InfiniBand (25 GB/s)
INFINIBAND_NDR_BW = 400 * Gbps            # NDR InfiniBand (50 GB/s)

# --- Energy (Horowitz, 2014 @ 45nm) ---
ENERGY_DRAM_ACCESS_PJ = 640 * ureg.picojoule
ENERGY_DRAM_PJ_PER_BYTE = 160 * ureg.picojoule / byte
ENERGY_FLOP_FP32_PJ = 3.7 * ureg.picojoule / flop   # FP32 multiply-add
ENERGY_FLOP_FP16_PJ = 1.1 * ureg.picojoule / flop   # FP16 multiply-add
ENERGY_FLOP_INT8_PJ = 0.2 * ureg.picojoule / flop   # INT8 multiply-add
ENERGY_FLOP_PJ = 4.6 * ureg.picojoule / flop         # Generic (legacy alias)
ENERGY_SRAM_L1_PJ = 0.5 * ureg.picojoule             # L1 cache access
ENERGY_SRAM_L2_PJ = 2.0 * ureg.picojoule             # L2 cache access
ENERGY_REG_PJ = 0.01 * ureg.picojoule                # Register file access
ENERGY_MOBILENET_INF_MJ = 0.1 * ureg.millijoule

# --- Physics ---
SPEED_OF_LIGHT_FIBER_KM_S = 200000 * ureg.kilometer / second

# --- Cloud Pricing ---
ureg.define('dollar = 1 * count')
USD = ureg.dollar
CLOUD_EGRESS_PER_GB = 0.09 * USD / GB
CLOUD_ELECTRICITY_PER_KWH = 0.12 * USD / ureg.kilowatt_hour

# --- Mobile / Battery ---
MOBILE_TDP_W = 3 * watt
PHONE_BATTERY_WH = 15 * watt * hour
OBJECT_DETECTOR_POWER_W = 2 * watt

# --- Video ---
VIDEO_1080P_WIDTH = 1920
VIDEO_1080P_HEIGHT = 1080
VIDEO_BYTES_PER_PIXEL_RGB = 3 * byte
VIDEO_FPS_STANDARD = Q_(30, 'Hz')

# --- Models & Workloads ---
ureg.define('param = 1 * count')
ureg.define('Mparam = 1e6 * param')

param = ureg.param
Mparam = ureg.Mparam

# GPT-2 (1.5B) — used in training chapter worked examples
GPT2_PARAMS = 1.5e9 * param
GPT2_LAYERS = 48
GPT2_HIDDEN_DIM = 1600

# GPT-3 (175B)
GPT3_PARAMS = 175e9 * param
GPT3_TRAINING_OPS = 3.14e23 * flop
GPT3_TRAINING_DAYS_REF = 25 # Days on 1024 A100s

# GPT-4 (Reference)
GPT4_TRAINING_GPU_DAYS = 2.5e6 # A100 days

# BERT-Base
BERT_BASE_PARAMS = 110e6 * param
BERT_BASE_FLOPs = 22e9 * flop              # Per inference (seq_len=512)

# Google Search (Reference)
GOOGLE_SEARCHES_PER_DAY = 8.5e9
GMAIL_EMAILS_PER_DAY = 121e9

# ResNet-50
RESNET50_PARAMS = 25.6e6 * param
RESNET50_FLOPs = 4.1e9 * flop

# MobileNetV2
MOBILENETV2_PARAMS = 3.5e6 * param
MOBILENETV2_FLOPs = 0.3e9 * flop

# YOLOv8-nano
YOLOV8_NANO_FLOPs = 3.2e9 * flop

# --- Storage (I/O Bandwidth) ---
NVME_SEQUENTIAL_BW = 3.5 * GB / second    # NVMe SSD sequential read
SYSTEM_MEMORY_BW = 50 * GB / second        # DDR4/DDR5 typical

# --- Case Studies ---
WAYMO_DATA_PER_HOUR_LOW = 1 * TB / hour
WAYMO_DATA_PER_HOUR_HIGH = 19 * TB / hour
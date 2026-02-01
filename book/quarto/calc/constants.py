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

# --- Time ---
MS = 1e-3 * second
US = 1e-6 * second
NS = 1e-9 * second

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

# NVIDIA V100 (Volta)
V100_FLOPS_FP16_TENSOR = 125 * TFLOPs / second
V100_FLOPS_FP32 = 15.7 * TFLOPs / second
V100_MEM_BW = 900 * GB / second
V100_MEM_CAPACITY = 32 * GiB

# NVIDIA A100 (Ampere)
A100_FLOPS_FP16_TENSOR = 312 * TFLOPs / second
A100_FLOPS_TF32 = 156 * TFLOPs / second
A100_MEM_BW = 2039 * GB / second
A100_MEM_CAPACITY = 80 * GiB

# NVIDIA H100 (Hopper)
H100_FLOPS_FP16_TENSOR = 989 * TFLOPs / second
H100_FLOPS_FP8_TENSOR = 1979 * TFLOPs / second
H100_MEM_BW = 3.35 * TB / second
H100_MEM_CAPACITY = 80 * GiB

# NVIDIA T4 (Turing)
T4_FLOPS_FP16_TENSOR = 65 * TFLOPs / second
T4_FLOPS_INT8 = 130 * TFLOPs / second
T4_MEM_BW = 320 * GB / second

# High-end Desktop CPU (Reference)
CPU_FLOPS_FP32 = 1 * TFLOPs / second

# Mobile NPU
MOBILE_NPU_TOPS_INT8 = 35 * TFLOPs / second
MOBILE_NPU_MEM_BW = 100 * GB / second

# --- Network ---
ureg.define('Gbps = 1e9 * bit / second')
Gbps = ureg.Gbps
NETWORK_10G_BW = 10 * Gbps
NETWORK_5G_ENERGY_PER_MB_MJ = 100 * ureg.millijoule / MB

# --- Energy ---
ENERGY_DRAM_ACCESS_PJ = 640 * ureg.picojoule       
ENERGY_DRAM_PJ_PER_BYTE = 160 * ureg.picojoule / byte
ENERGY_FLOP_PJ = 4.6 * ureg.picojoule / flop
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

# GPT-3 (175B)
GPT3_PARAMS = 175e9 * param
GPT3_TRAINING_OPS = 3.14e23 * flop
GPT3_TRAINING_DAYS_REF = 25 # Days on 1024 A100s

# GPT-4 (Reference)
GPT4_TRAINING_GPU_DAYS = 25000 # A100 days

# Google Search (Reference)
GOOGLE_SEARCHES_PER_DAY = 8.5e9

# ResNet-50
RESNET50_PARAMS = 25.6e6 * param
RESNET50_FLOPs = 4.1e9 * flop

# YOLOv8-nano
YOLOV8_NANO_FLOPs = 3.2e9 * flop

# --- Case Studies ---
WAYMO_DATA_PER_HOUR_LOW = 1 * TB / hour
WAYMO_DATA_PER_HOUR_HIGH = 19 * TB / hour
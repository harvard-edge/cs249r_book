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

KB = 1e3 * byte
MB = 1e6 * byte
GB = 1e9 * byte
TB = 1e12 * byte
PB = 1e15 * byte

KiB = 1024 * byte
MiB = 1024**2 * byte
GiB = 1024**3 * byte
TiB = 1024**4 * byte

# --- Time ---
MS = 1e-3 * second
US = 1e-6 * second
NS = 1e-9 * second

# --- Hardware Specifications (The Silicon Contract) ---

# FLOPs are dimensionless "operations"
ureg.define('flop = 1 * count')
flop = ureg.flop
TFLOPs = 1e12 * flop
ZFLOPs = 1e21 * flop

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
Gbps = 1e9 * ureg.bit / second
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

# --- Models & Workloads ---
ureg.define('param = 1 * count')
param = ureg.param

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
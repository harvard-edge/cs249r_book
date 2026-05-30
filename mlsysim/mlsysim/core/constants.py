# constants.py
#
# ⚠️ ARCHITECTURAL RULE ⚠️
# This module is now ONLY a re-export surface for measurement units (core/units.py).
# It holds NO domain values: every hardware/model/systems/literature/scenario figure
# has moved to its category registry, every physical constant to physics/constants.py,
# and every byte/bit width + PRECISION_MAP to core/units.py (the taxonomy refactor).
#
# DO NOT add a hardware/model/fleet/physics value here — it will hard-fail CI
# (mlsysim/tests/test_constants_allowlist.py). Homes:
#   Hardware specs        -> mlsysim/hardware/registry.py (+ Hardware.Tech.*)
#   Model specs           -> mlsysim/models/registry.py
#   Systems / fabrics     -> mlsysim/systems/registry.py
#   Case-study figures    -> mlsysim/scenarios/registry.py
#   Cited field figures   -> mlsysim/literature/registry.py
#   Formulas + physics    -> mlsysim/physics/ (incl. physics/constants.py)
#   Units / bit widths    -> mlsysim/core/units.py

from .units import *  # noqa: F401,F403 — re-export full unit registry

# --- System ratios / physics ---
# (Chip-specific numbers live in mlsysim/hardware/registry.py)

# --- Latency Hierarchy --- (tech-class access latency -> Hardware.Tech)
# register/L1/L2/HBM3 -> Hardware.Tech.Memory.*.latency
# NVLink/PCIe-Gen5    -> Hardware.Tech.Interconnect.*.latency
# NVMe SSD            -> Hardware.Tech.Storage.NvmeGen4.latency
# InfiniBand (fabric) -> Systems.Fabrics.InfiniBand_NDR.latency

# Mobile NPU peak power -> Scenarios.MobilePower.MobileNpuPeak

# Standard dimensions
# IMAGE_DIM_RESNET -> Datasets.ImageNet.image_width
# IMAGE_CHANNELS_RGB, COLOR_DEPTH_8BIT -> core/units.py (encoding facts)

# --- Network & Interconnect --- (network is a composition fact -> Systems)
# Ethernet 10G/100G/400G/800G/1.6T bandwidth -> Systems.Fabrics.Ethernet_*.bandwidth
# switch-ASIC capacity, 400G optics power, α/FEC/hop latencies -> Systems.SwitchFabric.*

# --- Energy --- (per-op / per-byte energy is a tech-class fact -> Hardware.Tech)
# FLOP/ADD/INT8 op energy   -> Hardware.Tech.Op.*.energy        (Horowitz 2014, 45 nm)
# DRAM access / per-byte    -> Hardware.Tech.Memory.HBM3.energy_per_access / energy_per_byte
# SRAM L1/L2, register file  -> Hardware.Tech.Memory.{L1,L2,Register}.energy_per_access
# Architecture-class efficiency + per-byte movement hierarchy -> Literature.Energy.*
# MobileNet inference energy -> Models.Vision.MobileNetV2.inference_energy

# Network transfer energy -> Systems.NetworkEnergy.{Per5gMb, Per1Kb}

# --- Physics --- (universal physical constants -> physics/constants.py)
# SPEED_OF_LIGHT_FIBER_KM_S -> physics/constants.py

# --- Mobile / Battery --- (device reference figures -> Scenarios)
# mobile NPU + object-detector power -> Scenarios.MobilePower.*
# flagship phone battery (Wh) -> Scenarios.PhoneBattery.EnergyWh

# Reference energy-scale anchors -> Scenarios.EnergyAnchors.{SmartphoneCharge,BoilingWater}

# --- Video --- (VIDEO_* encoding/format facts -> core/units.py)

# Reference model/dataset dimensions
# TRANSFORMER FLOP ratios -> Literature.Chinchilla.{ComputeConstant(6PD), DecodeConstant(2P)}
# TRANSFORMER_*_EXAMPLE dims -> inlined in the one worked example that used them (hw_acceleration)
# SIMD_REGISTER_BITS, FP32_BITS, INT8_BITS -> core/units.py (bit widths)
# Single-chapter pedagogical example scalars are inlined in their LEGO cells (no
# canonical registry home, per the architecture's pedagogical-input carve-out):
#   SYSTOLIC_ARRAY_DIM -> hw_acceleration; SYNTHETIC_* -> data_storage;
#   LOGIC_WALL_REASONING_STEPS_EXAMPLE -> inference; ML_WORKFLOW_STAGE_*/COST_BASE -> ml_workflow
# GOOGLE_SEARCHES_PER_DAY, GMAIL_EMAILS_PER_DAY -> Scenarios.Workloads

# --- Storage (I/O Bandwidth) ---
# NVME_*/SYSTEM_MEMORY_BW/HOST_DRAM_BW -> Hardware.Tech.Storage (tech-class bandwidth)
# LOCAL_NVME_DRIVES_PER_NODE_REF -> inlined node-config in data_storage (pedagogical)

# --- Case Studies --- (WAYMO_*, ANOMALY_MODEL_* -> Scenarios.Workloads / Scenarios.AnomalyModel)

# Phone battery capacity/voltage/energy -> Scenarios.PhoneBattery.{CapacityMah,VoltageV,EnergyJ}

# PRECISION_MAP -> core/units.py (precision-string -> byte-width is a measurement fact)


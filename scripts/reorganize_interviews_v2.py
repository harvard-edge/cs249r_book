#!/usr/bin/env python3
"""
Reorganize ML Systems Interview Playbook: scope-first, competency-second.

Files are organized by the SYSTEM the learner is reasoning about (scope),
and within each file, questions are grouped by competency topic, then sorted
by mastery level (L3 → L6+).

Usage:
    python3 scripts/reorganize_interviews_v2.py [--dry-run]
"""

import re
import os
import sys
import shutil
import argparse
from collections import defaultdict, OrderedDict
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent / "interviews"
TRACKS = ["cloud", "edge", "mobile", "tinyml"]
TRACK_DISPLAY = {"cloud": "Cloud", "edge": "Edge", "mobile": "Mobile", "tinyml": "TinyML"}

LEVEL_ORDER = {"L3": 0, "L4": 1, "L5": 2, "L6+": 3}

# ─────────────────────────────────────────────────────────────────────
# Scope definitions per track
# ─────────────────────────────────────────────────────────────────────
# Each scope = one output file. Within each scope, questions are grouped
# by competency sub-sections. The learner opens a file based on "what
# system am I studying?" and finds organized topics inside.

# For each scope: (slug, title, subtitle, description, competency_sections)
# competency_sections = ordered list of (section_title, section_emoji)

TRACK_SCOPES = {
    "cloud": [
        {
            "slug": "01_single_machine",
            "title": "The Single Machine",
            "subtitle": "What happens inside one server",
            "description": "Roofline analysis, memory hierarchies, numerical precision, hardware architecture, and data pipelines — everything that determines performance within a single node.",
            "sections": [
                ("Roofline & Compute Analysis", "📐"),
                ("Memory Hierarchy & KV-Cache", "🧠"),
                ("Numerical Precision & Quantization", "🔢"),
                ("Hardware Architecture & Cost", "🏗️"),
                ("Compilers & Frameworks", "⚙️"),
                ("Data Pipelines", "📊"),
            ],
        },
        {
            "slug": "02_distributed_systems",
            "title": "The Distributed System",
            "subtitle": "What happens when you exceed one node",
            "description": "Parallelism strategies, network topology, collective communication, and fault tolerance — the physics of keeping thousands of GPUs fed and synchronized.",
            "sections": [
                ("Parallelism & Memory Sharding", "🔀"),
                ("Network Topology & Collectives", "🌐"),
                ("Fault Tolerance & Reliability", "🛡️"),
                ("Training at Scale", "🏋️"),
            ],
        },
        {
            "slug": "03_serving_stack",
            "title": "The Serving Stack",
            "subtitle": "How you serve models to real users",
            "description": "Latency budgets, batching strategies, KV-cache management, autoscaling, and speculative decoding — surviving real user traffic at scale.",
            "sections": [
                ("Latency & Throughput", "⏱️"),
                ("Batching & Scheduling", "🔄"),
                ("KV-Cache & Memory Management", "🗂️"),
                ("Serving Architecture", "🏗️"),
                ("Advanced Inference", "🚀"),
            ],
        },
        {
            "slug": "04_production_ops",
            "title": "The Production System",
            "subtitle": "How you keep it running and healthy",
            "description": "Monitoring, drift detection, deployment strategies, security, power management, and economics — the operational reality of ML in production.",
            "sections": [
                ("Monitoring & Observability", "📉"),
                ("Deployment & MLOps", "🚀"),
                ("Data Quality & Pipelines", "📊"),
                ("Power, Thermal & Sustainability", "⚡"),
                ("Economics & Infrastructure", "💰"),
                ("Security, Privacy & Fairness", "🔒"),
            ],
        },
        {
            "slug": "05_visual_debugging",
            "title": "Visual Architecture Debugging",
            "subtitle": "Can you spot the bottleneck in a diagram?",
            "description": "System architecture diagrams with hidden bottlenecks. Read the diagram, find the constraint, and explain the fix.",
            "sections": [],  # Visual debugging has its own format
        },
    ],
    "edge": [
        {
            "slug": "01_hardware_platform",
            "title": "The Hardware Platform",
            "subtitle": "What silicon are you working with and what are its limits?",
            "description": "Edge accelerator rooflines, memory hierarchies, numerical precision, SoC architectures, and heterogeneous compute — understanding the hardware constraints of edge deployment.",
            "sections": [
                ("Roofline & Compute Analysis", "📐"),
                ("Memory Systems", "🧠"),
                ("Numerical Precision & Quantization", "🔢"),
                ("Architecture & Heterogeneous Compute", "🏗️"),
            ],
        },
        {
            "slug": "02_realtime_pipeline",
            "title": "The Real-Time Pipeline",
            "subtitle": "How you meet deadlines with sensor data",
            "description": "Real-time scheduling, sensor fusion, latency budgets, power management, and thermal constraints — the physics of processing sensor data under hard deadlines.",
            "sections": [
                ("Real-Time & Latency", "⏱️"),
                ("Sensor Fusion & Pipelines", "📡"),
                ("Power & Thermal Management", "⚡"),
                ("Model Optimization", "🔧"),
            ],
        },
        {
            "slug": "03_deployed_system",
            "title": "The Deployed System",
            "subtitle": "How you get it into the field and keep it running",
            "description": "OTA updates, fleet management, monitoring, functional safety, security, and long-term reliability — operating ML at the edge of the network.",
            "sections": [
                ("Deployment & Fleet Management", "🚀"),
                ("Monitoring & Reliability", "🛡️"),
                ("Functional Safety", "🔒"),
                ("Security & Privacy", "🔐"),
                ("Economics & Long-Term Operations", "💰"),
            ],
        },
        {
            "slug": "04_visual_debugging",
            "title": "Visual Architecture Debugging",
            "subtitle": "Can you spot the bottleneck in an edge system diagram?",
            "description": "Edge system architecture diagrams with hidden bottlenecks.",
            "sections": [],
        },
    ],
    "mobile": [
        {
            "slug": "01_device_hardware",
            "title": "The Device & SoC",
            "subtitle": "What hardware are you working with?",
            "description": "SoC architecture, NPU delegation, memory hierarchies, numerical precision, and heterogeneous compute — understanding the mobile hardware stack from CPU to NPU.",
            "sections": [
                ("Compute & SoC Architecture", "📐"),
                ("Memory Systems", "🧠"),
                ("Numerical Precision & Quantization", "🔢"),
                ("NPU, GPU & Heterogeneous Compute", "🏗️"),
            ],
        },
        {
            "slug": "02_app_experience",
            "title": "The App Experience",
            "subtitle": "How you make inference feel instant",
            "description": "Latency budgets, UI jank, thermal throttling, power management, compiler optimization, and model optimization — making ML invisible to the user.",
            "sections": [
                ("Latency & Responsiveness", "⏱️"),
                ("Power & Thermal Management", "⚡"),
                ("Compilers & Frameworks", "⚙️"),
                ("Model Optimization", "🔧"),
                ("Sensor & Media Pipelines", "📡"),
            ],
        },
        {
            "slug": "03_ship_and_update",
            "title": "Ship & Update",
            "subtitle": "How you ship models to a billion phones and keep them current",
            "description": "App store constraints, model delivery, A/B testing, monitoring, privacy, and on-device training — the lifecycle of ML on mobile devices.",
            "sections": [
                ("Deployment & Model Delivery", "🚀"),
                ("Monitoring & Reliability", "🛡️"),
                ("Privacy & Security", "🔒"),
                ("On-Device Training & Federated Learning", "🏋️"),
                ("Economics & Platform Constraints", "💰"),
            ],
        },
        {
            "slug": "04_visual_debugging",
            "title": "Visual Architecture Debugging",
            "subtitle": "Can you spot the bottleneck in a mobile system diagram?",
            "description": "Mobile system architecture diagrams with hidden bottlenecks.",
            "sections": [],
        },
    ],
    "tinyml": [
        {
            "slug": "01_microcontroller",
            "title": "The Microcontroller",
            "subtitle": "What fits in 256 KB of SRAM?",
            "description": "MCU architectures, SRAM partitioning, flash storage, integer arithmetic, instruction sets, and compiler optimization — the extreme constraints of microcontroller ML.",
            "sections": [
                ("Compute & Architecture", "📐"),
                ("Memory Systems & Flash", "🧠"),
                ("Numerical Precision & Quantization", "🔢"),
                ("Compilers & Frameworks", "⚙️"),
            ],
        },
        {
            "slug": "02_sensing_pipeline",
            "title": "The Sensing Pipeline",
            "subtitle": "From sensor input to inference output",
            "description": "Sensor interfaces, real-time scheduling, power management, duty cycling, and model optimization — processing sensor data under extreme resource constraints.",
            "sections": [
                ("Real-Time & Latency", "⏱️"),
                ("Sensor Pipelines", "📡"),
                ("Power & Energy Management", "⚡"),
                ("Model Optimization", "🔧"),
            ],
        },
        {
            "slug": "03_deployed_device",
            "title": "The Deployed Device",
            "subtitle": "How you update firmware and keep it alive for years",
            "description": "FOTA updates, connectivity, monitoring, security, and long-term reliability — operating ML on devices that must run unattended for years.",
            "sections": [
                ("Deployment & Updates", "🚀"),
                ("Networking & Connectivity", "🌐"),
                ("Monitoring & Reliability", "🛡️"),
                ("Security & Privacy", "🔒"),
                ("Economics & Hardware Design", "💰"),
            ],
        },
        {
            "slug": "04_visual_debugging",
            "title": "Visual Architecture Debugging",
            "subtitle": "Can you spot the bottleneck in a TinyML system diagram?",
            "description": "TinyML system architecture diagrams with hidden bottlenecks.",
            "sections": [],
        },
    ],
}

# ─────────────────────────────────────────────────────────────────────
# Tag → Scope mapping (per track)
# ─────────────────────────────────────────────────────────────────────
# Maps primary tags to scope slugs. If a tag appears in multiple scopes,
# the first match wins.

SCOPE_TAGS = {
    "cloud": {
        "01_single_machine": {
            "roofline", "compute", "cpu-gpu-arch", "tpu-architecture", "cpu-architecture",
            "compute-intensity", "compute-overhead", "cpu",
            "memory-hierarchy", "memory", "memory-bandwidth", "kv-cache",
            "cache-hierarchy", "cache-coherence", "virtual-memory", "data-locality",
            "NUMA", "numa-multicore", "memory-technologies", "memory-pressure",
            "memory-layout", "memory-management", "memory-footprint",
            "memory-technologies", "memory-bus", "memory-architecture",
            "precision", "numerical-precision", "quantization", "mixed-precision",
            "architecture", "hardware", "sparsity", "custom-hardware",
            "hardware-topology", "cpu-architecture",
            "compiler-runtime", "frameworks",
            "data-pipeline", "data-loading",
            "gpu",
        },
        "02_distributed_systems": {
            "parallelism", "model-parallelism",
            "network-fabric", "network", "network-topology", "network-io",
            "network-management", "network-protocol", "networking",
            "heterogeneous-network", "cross-region", "wan-optimization",
            "collectives", "collective-communication", "collective-design",
            "bandwidth", "topology",
            "fault-tolerance", "reproducibility",
            "training", "distributed-training", "distributed",
        },
        "03_serving_stack": {
            "serving", "latency", "throughput", "queueing", "queueing-theory",
            "batching", "continuous-batching", "scheduling",
            "speculative-decoding", "attention-mechanisms",
            "kv-cache-management", "serverless-inference", "llm-serving",
            "inference",
        },
        "04_production_ops": {
            "mlops", "monitoring", "model-monitoring",
            "deployment", "canary-deployment", "containerization",
            "economics", "cost-optimization", "cost-attribution", "spot-instances",
            "power-thermal", "power", "energy", "sustainability", "sustainable-ai",
            "carbon-aware", "datacenter-ops",
            "security", "privacy", "fairness",
            "data-quality", "data-versioning", "data-consistency", "data-contracts",
            "data-tiering", "data-lifecycle", "data-privacy", "data-provenance",
            "feature-store", "real-time-feature-store", "real-time-ml",
            "model-lifecycle", "model-monitoring",
            "storage", "storage-io", "io-optimization",
            "health-checks",
            "incident-response",
            "artifact-management", "workflow-orchestration",
            "streaming", "cluster-scheduling", "resource-scheduling",
        },
    },
    "edge": {
        "01_hardware_platform": {
            "roofline", "compute", "cpu", "gpu",
            "memory-hierarchy", "memory", "memory-bandwidth", "kv-cache",
            "memory-footprint", "on-chip-memory", "scratchpad-memory",
            "flash-memory", "memory-mapped-io", "memory-capacity",
            "memory-power", "memory-bus", "shared-bandwidth",
            "dynamic-memory-allocation", "dma", "dma-transfers", "mmio",
            "cache-performance", "numa-multicore",
            "precision", "quantization", "quantization-memory",
            "quantization-robustness", "mixed-precision",
            "architecture", "hardware", "heterogeneous-compute",
            "system-on-chip", "custom-hardware", "hardware-acceleration",
            "sparsity-memory",
            "heterogeneous-memory-coherence",
        },
        "02_realtime_pipeline": {
            "real-time", "latency", "real-time-latency", "real-time-systems",
            "timing", "wcet", "rtos", "rtos-deterministic", "deterministic-timing",
            "scheduling", "os", "concurrency",
            "sensor-pipeline", "sensor-fusion", "multi-sensor-fusion",
            "sensor", "sensors", "sensor-calibration", "sensor-interface",
            "sensor-io", "sensor-physics", "vision",
            "power-thermal", "power", "power-management", "power-management-adaptive",
            "power-gating", "ultra-low-power", "thermal", "thermal-management",
            "thermal-throttling",
            "compiler-runtime", "optimization", "model-optimization",
            "model-compression",
        },
        "03_deployed_system": {
            "deployment", "edge-deployment", "ota-updates", "ota",
            "model-update", "model-versioning", "model-deployment",
            "monitoring", "reliability", "long-term-reliability",
            "watchdog", "interrupt-vs-polling", "state-machine", "boot-time",
            "functional-safety", "safety",
            "security", "privacy", "adversarial-robustness",
            "physical-security", "secure-boot",
            "hardware-root-of-trust", "attestation", "ip-protection",
            "supply-chain-security",
            "economics", "hardware-lifecycle", "long-term-autonomy",
            "mlops", "fleet-management", "edge-cloud-sync",
            "data-pipeline", "data-management", "data-collection",
            "data-quality", "data-provenance",
            "robustness", "self-healing",
            "a/b-testing", "ci/cd", "feature-flags", "rollout-strategies",
            "anomaly-detection", "connectivity", "offline", "offline-operations",
            "bandwidth-constraints", "resource-constraints", "heterogeneity",
            "multi-tenant", "multi-model",
            "network-fabric", "networking", "storage",
            "pipeline",
            "sensor-pipeline",  # when in deployment context
        },
    },
    "mobile": {
        "01_device_hardware": {
            "compute", "compute-intensity", "compute-overhead",
            "compute-memory-bandwidth", "cpu-architecture",
            "memory-hierarchy", "memory", "memory-bandwidth", "memory-pressure",
            "fragmentation", "memory-layout", "memory-management",
            "memory-architecture", "memory-footprint",
            "memory-management-oom-android", "memory-management-latency-spikes-android",
            "cpu-cache-memory-access", "kv-cache",
            "precision", "numerical-precision", "quantization", "mixed-precision",
            "dynamic-quantization", "quantization-hardware", "quantization-npu",
            "quantization-memory-deployment",
            "quantization-low-precision-hardware-support",
            "architecture", "hardware", "soc", "soc-architecture", "soc-interconnect",
            "npu-delegation", "npu-architecture", "npu-compiler",
            "heterogeneous-compute", "apple-silicon",
            "custom-ops-heterogeneous-compute-vendor-sdk",
            "gpu", "mobile-gpu", "roofline",
        },
        "02_app_experience": {
            "latency", "latency-budgets", "cold-start-latency",
            "display-pipeline", "resource-contention",
            "heterogeneous-compute-scheduling-latency-power",
            "power-thermal", "power", "power-management", "power-consumption",
            "battery", "battery-impact", "dvfs",
            "thermal-management", "thermal-throttling", "thermal-design",
            "thermal-power-sustained-perf",
            "power-efficiency-frameworks-android-ios",
            "compiler-runtime", "compiler", "frameworks",
            "optimization", "profiling", "performance",
            "cpu-npu-handoff", "zero-copy-pipeline",
            "scheduling", "os-scheduling", "os", "concurrency",
            "pipeline", "sensor-pipeline", "sensor-fusion",
            "sensors", "vision", "audio",
            "model-compression",
        },
        "03_ship_and_update": {
            "deployment", "model-delivery", "model-loading", "model-formats",
            "model-versioning", "app-lifecycle",
            "serving",
            "monitoring", "mlops",
            "privacy", "security", "fairness",
            "training", "on-device-training", "distributed",
            "federated-learning",
            "testing", "platform",
            "economics",
            "storage", "storage-io",
            "data-pipeline",
            "reliability",
        },
    },
    "tinyml": {
        "01_microcontroller": {
            "compute", "cpu", "gpu",
            "instruction-set", "branch-prediction", "instruction-cache",
            "hardware-mac", "bus-arbitration", "clock-tree", "debug-interface",
            "boot-sequence", "peripheral-timer",
            "memory-hierarchy", "memory", "memory-layout", "memory-alignment",
            "memory-mapped-io", "flash-memory", "on-chip-memory",
            "dma", "memory-capacity",
            "precision", "quantization", "integer-inference", "math",
            "compiler-runtime", "compiler", "frameworks",
            "architecture", "hardware", "model-architecture",
            "roofline",
        },
        "02_sensing_pipeline": {
            "real-time", "latency", "timing", "rtos-scheduling", "concurrency",
            "sensor-pipeline", "sensor-fusion", "sensor", "sensors",
            "sensor-interface", "sensor-io", "vision", "audio",
            "power-thermal", "power", "power-energy",
            "energy-harvesting", "voltage-scaling", "low-power",
            "optimization", "model-compression", "performance",
            "pipeline",
            "heterogeneous-compute", "micro-npu", "hw-acceleration",
            "sparsity",
        },
        "03_deployed_device": {
            "deployment", "storage",
            "networking", "network-fabric",
            "monitoring", "reliability", "watchdog",
            "functional-safety", "robustness",
            "security", "privacy",
            "mlops", "training",
            "data-pipeline", "data-management",
            "system-design", "nas",
            "parallelism",
            "model-optimization",
        },
    },
}

# ─────────────────────────────────────────────────────────────────────
# Competency sub-section mapping (tag → section title within a scope)
# ─────────────────────────────────────────────────────────────────────
# Used to assign questions to the right sub-section within their scope file.

COMPETENCY_LABELS = {
    # Compute
    "roofline": "Roofline & Compute Analysis",
    "compute": "Roofline & Compute Analysis",
    "cpu-gpu-arch": "Roofline & Compute Analysis",
    "tpu-architecture": "Roofline & Compute Analysis",
    "cpu-architecture": "Roofline & Compute Analysis",
    "compute-intensity": "Roofline & Compute Analysis",
    "compute-overhead": "Roofline & Compute Analysis",
    "compute-memory-bandwidth": "Compute & SoC Architecture",
    "cpu": "Roofline & Compute Analysis",
    "gpu": "Roofline & Compute Analysis",
    "instruction-set": "Compute & Architecture",
    "branch-prediction": "Compute & Architecture",
    "instruction-cache": "Compute & Architecture",
    "hardware-mac": "Compute & Architecture",
    "bus-arbitration": "Compute & Architecture",
    "clock-tree": "Compute & Architecture",
    "debug-interface": "Compute & Architecture",
    "boot-sequence": "Compute & Architecture",
    "peripheral-timer": "Compute & Architecture",

    # Memory
    "memory-hierarchy": "Memory Systems",
    "memory": "Memory Systems",
    "memory-bandwidth": "Memory Systems",
    "kv-cache": "Memory Hierarchy & KV-Cache",
    "kv-cache-management": "KV-Cache & Memory Management",
    "cache-hierarchy": "Memory Systems",
    "cache-coherence": "Memory Systems",
    "cache-performance": "Memory Systems",
    "virtual-memory": "Memory Systems",
    "data-locality": "Memory Systems",
    "NUMA": "Memory Systems",
    "numa-multicore": "Memory Systems",
    "memory-technologies": "Memory Systems",
    "memory-pressure": "Memory Systems",
    "fragmentation": "Memory Systems",
    "memory-layout": "Memory Systems & Flash",
    "memory-management": "Memory Systems",
    "memory-footprint": "Memory Systems",
    "on-chip-memory": "Memory Systems",
    "scratchpad-memory": "Memory Systems",
    "flash-memory": "Memory Systems & Flash",
    "memory-mapped-io": "Memory Systems & Flash",
    "memory-alignment": "Memory Systems & Flash",
    "memory-capacity": "Memory Systems",
    "memory-power": "Memory Systems",
    "memory-bus": "Memory Systems",
    "memory-architecture": "Memory Systems",
    "dma": "Memory Systems",
    "dma-transfers": "Memory Systems",
    "mmio": "Memory Systems",
    "shared-bandwidth": "Memory Systems",
    "dynamic-memory-allocation": "Memory Systems",
    "storage": "Memory Systems",
    "storage-io": "Memory Systems",

    # Precision
    "precision": "Numerical Precision & Quantization",
    "numerical-precision": "Numerical Precision & Quantization",
    "quantization": "Numerical Precision & Quantization",
    "mixed-precision": "Numerical Precision & Quantization",
    "integer-inference": "Numerical Precision & Quantization",
    "math": "Numerical Precision & Quantization",

    # Architecture
    "architecture": "Hardware Architecture & Cost",
    "hardware": "Hardware Architecture & Cost",
    "soc": "NPU, GPU & Heterogeneous Compute",
    "soc-architecture": "NPU, GPU & Heterogeneous Compute",
    "soc-interconnect": "NPU, GPU & Heterogeneous Compute",
    "heterogeneous-compute": "Architecture & Heterogeneous Compute",
    "npu-delegation": "NPU, GPU & Heterogeneous Compute",
    "npu-architecture": "NPU, GPU & Heterogeneous Compute",
    "npu-compiler": "NPU, GPU & Heterogeneous Compute",
    "apple-silicon": "NPU, GPU & Heterogeneous Compute",
    "model-compression": "Model Optimization",
    "sparsity": "Hardware Architecture & Cost",
    "hardware-topology": "Hardware Architecture & Cost",
    "model-architecture": "Hardware Architecture & Cost",

    # Parallelism & Network
    "parallelism": "Parallelism & Memory Sharding",
    "model-parallelism": "Parallelism & Memory Sharding",
    "network-fabric": "Network Topology & Collectives",
    "network": "Network Topology & Collectives",
    "network-topology": "Network Topology & Collectives",
    "collectives": "Network Topology & Collectives",
    "collective-communication": "Network Topology & Collectives",
    "collective-design": "Network Topology & Collectives",
    "bandwidth": "Network Topology & Collectives",
    "topology": "Network Topology & Collectives",
    "fault-tolerance": "Fault Tolerance & Reliability",

    # Training
    "training": "Training at Scale",
    "distributed-training": "Training at Scale",
    "distributed": "Training at Scale",

    # Serving
    "serving": "Serving Architecture",
    "latency": "Latency & Throughput",
    "throughput": "Latency & Throughput",
    "queueing": "Latency & Throughput",
    "queueing-theory": "Latency & Throughput",
    "batching": "Batching & Scheduling",
    "continuous-batching": "Batching & Scheduling",
    "scheduling": "Batching & Scheduling",
    "speculative-decoding": "Advanced Inference",
    "attention-mechanisms": "Advanced Inference",

    # Real-time
    "real-time": "Real-Time & Latency",
    "timing": "Real-Time & Latency",
    "wcet": "Real-Time & Latency",
    "rtos": "Real-Time & Latency",
    "rtos-scheduling": "Real-Time & Latency",

    # Sensor
    "sensor-pipeline": "Sensor Fusion & Pipelines",
    "sensor-fusion": "Sensor Fusion & Pipelines",
    "sensor": "Sensor Fusion & Pipelines",
    "sensors": "Sensor & Media Pipelines",

    # Power
    "power-thermal": "Power & Thermal Management",
    "power": "Power, Thermal & Sustainability",
    "power-management": "Power & Thermal Management",
    "thermal": "Power & Thermal Management",
    "thermal-management": "Power & Thermal Management",
    "thermal-throttling": "Power & Thermal Management",
    "energy": "Power, Thermal & Sustainability",
    "battery": "Power & Thermal Management",
    "sustainability": "Economics & Infrastructure",
    "energy-harvesting": "Power & Energy Management",
    "voltage-scaling": "Power & Energy Management",

    # MLOps & Production
    "mlops": "Deployment & MLOps",
    "monitoring": "Monitoring & Observability",
    "deployment": "Deployment & Fleet Management",
    "economics": "Economics & Infrastructure",
    "security": "Security & Privacy",
    "privacy": "Privacy & Security",
    "fairness": "Security, Privacy & Fairness",
    "incident-response": None,  # Assigned to parent scope's most relevant section
    "data-pipeline": "Data Quality & Pipelines",
    "data-quality": "Data Quality & Pipelines",
    "feature-store": "Data Quality & Pipelines",

    # Compilers
    "compiler-runtime": "Compilers & Frameworks",
    "frameworks": "Compilers & Frameworks",
    "compiler": "Compilers & Frameworks",
    "optimization": "Model Optimization",
    "profiling": "Compilers & Frameworks",
    "performance": "Model Optimization",

    # Reliability
    "reliability": "Monitoring & Reliability",
    "functional-safety": "Functional Safety",
    "robustness": "Monitoring & Reliability",
    "safety": "Functional Safety",
    "watchdog": "Monitoring & Reliability",

    # Connectivity
    "networking": "Networking & Connectivity",
}

LEVEL_HEADERS = {
    "L3": "#### 🟢 L3 — Recall & Define",
    "L4": "#### 🔵 L4 — Apply & Identify",
    "L5": "#### 🟡 L5 — Analyze & Predict",
    "L6+": "#### 🔴 L6+ — Synthesize & Derive",
}


# ─────────────────────────────────────────────────────────────────────
# Parser (same as v1)
# ─────────────────────────────────────────────────────────────────────

def extract_questions(filepath):
    """Extract all <details> question blocks from a markdown file."""
    with open(filepath, "r") as f:
        content = f.read()

    questions = []
    i = 0
    while i < len(content):
        start = content.find("<details>", i)
        if start == -1:
            break

        depth = 0
        j = start
        end = -1
        while j < len(content):
            open_tag = content.find("<details>", j)
            close_tag = content.find("</details>", j)
            if close_tag == -1:
                break
            if open_tag != -1 and open_tag < close_tag:
                depth += 1
                j = open_tag + len("<details>")
            else:
                depth -= 1
                if depth == 0:
                    end = close_tag + len("</details>")
                    break
                j = close_tag + len("</details>")

        if end == -1:
            i = start + 1
            continue

        block = content[start:end]
        summary_match = re.search(r'<summary>(.*?)</summary>', block, re.DOTALL)
        if not summary_match:
            i = end
            continue

        summary = summary_match.group(1)
        level_match = re.search(r'Level-(\w+)', summary)
        if not level_match:
            i = end
            continue

        level_raw = level_match.group(1).split('_')[0]
        if "Principal" in level_match.group(1) or level_raw == "L6":
            level = "L6+"
        else:
            level = level_raw

        title_match = re.search(r'>\s*([^<]+)</b>', summary)
        title = title_match.group(1).strip() if title_match else "Untitled"
        tags = re.findall(r'<code>([^<]+)</code>', summary)

        questions.append({
            "title": title,
            "level": level,
            "tags": tags,
            "body": block,
            "source_file": os.path.basename(filepath),
        })
        i = end

    return questions


def classify_scope(question, track):
    """Assign a question to a scope based on its tags and track."""
    tags = question["tags"]
    scope_tags = SCOPE_TAGS[track]

    # Try each scope in order (priority: earlier scopes win)
    for scope_slug, tag_set in scope_tags.items():
        if scope_slug.endswith("visual_debugging"):
            continue  # visual handled separately
        for tag in tags:
            if tag in tag_set:
                return scope_slug

    # Fallback: last non-visual scope
    scopes = [s for s in scope_tags if not s.endswith("visual_debugging")]
    return scopes[-1]


def get_section_label(question, scope_sections):
    """Get the competency sub-section label for a question.

    Uses explicit mapping first, then keyword matching against
    the scope's defined sections as fallback.
    """
    tags = question["tags"]
    section_names = [s[0] for s in scope_sections] if scope_sections else []

    # 1. Try explicit mapping
    for tag in tags:
        if tag in COMPETENCY_LABELS and COMPETENCY_LABELS[tag] is not None:
            return COMPETENCY_LABELS[tag]

    # 2. Try keyword matching: split tag on hyphens and match against section names
    for tag in tags:
        tag_words = set(tag.lower().replace("-", " ").replace("_", " ").split())
        best_score = 0
        best_section = None
        for sec_name in section_names:
            sec_words = set(sec_name.lower().replace("&", "").replace(",", "").split())
            overlap = len(tag_words & sec_words)
            if overlap > best_score:
                best_score = overlap
                best_section = sec_name
        if best_score > 0:
            return best_section

    # 3. Use first section as default (better than "Additional Topics")
    if section_names:
        return section_names[0]

    return "Additional Topics"


def level_sort_key(level):
    return LEVEL_ORDER.get(level, 99)


# ─────────────────────────────────────────────────────────────────────
# Track metadata
# ─────────────────────────────────────────────────────────────────────

TRACK_META = {
    "cloud": {
        "emoji": "☁️",
        "name": "Cloud Track — Data Center & Distributed Systems",
        "description": "The Cloud track covers ML systems that run in data centers — from a single H100 to 10,000-GPU training clusters to production serving fleets handling millions of requests per second.",
        "constraint_table": """<table>
  <thead><tr><th width="25%">Dimension</th><th width="75%">Cloud Reality</th></tr></thead>
  <tbody>
    <tr><td><b>Compute</b></td><td>PFLOPS (H100, TPU, B200)</td></tr>
    <tr><td><b>Memory</b></td><td>80 GB HBM per chip, terabytes across a cluster</td></tr>
    <tr><td><b>Interconnect</b></td><td>NVLink (900 GB/s intra-node), InfiniBand (400 Gbps inter-node)</td></tr>
    <tr><td><b>Power budget</b></td><td>700W–1000W per chip, megawatts per cluster</td></tr>
    <tr><td><b>Primary bottleneck</b></td><td>Memory bandwidth (single node), network (multi-node)</td></tr>
    <tr><td><b>Failure mode</b></td><td>Silent data corruption at scale, straggler nodes, MTBF collapse</td></tr>
  </tbody>
</table>""",
        "audience": "Engineers interviewing at frontier labs and cloud infrastructure companies — Meta, Google, OpenAI, Anthropic, NVIDIA, Amazon, Microsoft, and similar organizations building or operating large-scale ML systems.",
    },
    "edge": {
        "emoji": "🤖",
        "name": "Edge Track — Autonomous Systems & Industrial AI",
        "description": "The Edge track covers ML systems deployed on dedicated hardware at the point of action — autonomous vehicles, robotics platforms, CCTV and surveillance systems, industrial inspection, and medical devices.",
        "constraint_table": """<table>
  <thead><tr><th width="25%">Dimension</th><th width="75%">Edge Reality</th></tr></thead>
  <tbody>
    <tr><td><b>Compute</b></td><td>TOPS (Jetson Orin, Hailo-8, Intel Movidius, Google Coral)</td></tr>
    <tr><td><b>Memory</b></td><td>8–32 GB DRAM, shared with sensor pipelines</td></tr>
    <tr><td><b>Interconnect</b></td><td>PCIe, MIPI CSI (camera), CAN bus (automotive)</td></tr>
    <tr><td><b>Power budget</b></td><td>15–75W per module</td></tr>
    <tr><td><b>Primary bottleneck</b></td><td>Thermal envelope and real-time deadlines</td></tr>
    <tr><td><b>Failure mode</b></td><td>Missing a hard real-time deadline, thermal throttling under sustained load</td></tr>
  </tbody>
</table>""",
        "audience": "Engineers interviewing at autonomous vehicle companies (Tesla, Waymo, Cruise), robotics firms (Boston Dynamics, Agility), industrial AI startups, and edge computing platforms (NVIDIA, Qualcomm, Hailo).",
    },
    "mobile": {
        "emoji": "📱",
        "name": "Mobile Track — On-Device AI for Smartphones",
        "description": "The Mobile track covers ML systems that run on smartphones and tablets — on-device inference, NPU delegation, app store constraints, and battery-aware optimization.",
        "constraint_table": """<table>
  <thead><tr><th width="25%">Dimension</th><th width="75%">Mobile Reality</th></tr></thead>
  <tbody>
    <tr><td><b>Compute</b></td><td>TOPS (Snapdragon, Apple Neural Engine, MediaTek APU, Samsung NPU)</td></tr>
    <tr><td><b>Memory</b></td><td>6–12 GB shared with OS and apps, no dedicated VRAM</td></tr>
    <tr><td><b>Interconnect</b></td><td>On-SoC fabric, shared memory bus</td></tr>
    <tr><td><b>Power budget</b></td><td>3–5W total device power</td></tr>
    <tr><td><b>Primary bottleneck</b></td><td>Battery life and shared resources</td></tr>
    <tr><td><b>Failure mode</b></td><td>Thermal throttling, app eviction, silent accuracy loss</td></tr>
  </tbody>
</table>""",
        "audience": "Engineers interviewing at Apple, Google, Samsung, Qualcomm, and mobile-first AI companies building on-device ML features.",
    },
    "tinyml": {
        "emoji": "🔬",
        "name": "TinyML Track — Microcontroller & Ultra-Low-Power AI",
        "description": "The TinyML track covers ML systems that run on microcontrollers and ultra-low-power devices — always-on sensing, energy harvesting, hard real-time inference in kilobytes of SRAM.",
        "constraint_table": """<table>
  <thead><tr><th width="25%">Dimension</th><th width="75%">TinyML Reality</th></tr></thead>
  <tbody>
    <tr><td><b>Compute</b></td><td>MFLOPS (Cortex-M, RISC-V, custom accelerators)</td></tr>
    <tr><td><b>Memory</b></td><td>256 KB – 2 MB SRAM, 1–16 MB flash</td></tr>
    <tr><td><b>Interconnect</b></td><td>SPI, I2C, UART, GPIO</td></tr>
    <tr><td><b>Power budget</b></td><td>Microwatts to milliwatts</td></tr>
    <tr><td><b>Primary bottleneck</b></td><td>SRAM capacity and hard real-time deadlines</td></tr>
    <tr><td><b>Failure mode</b></td><td>Model doesn't fit in SRAM, missed interrupt deadlines, energy budget exceeded</td></tr>
  </tbody>
</table>""",
        "audience": "Engineers interviewing at sensor companies, IoT platforms, wearable tech firms, and embedded AI startups deploying ML to devices that run on batteries or harvested energy.",
    },
}


# ─────────────────────────────────────────────────────────────────────
# Writers
# ─────────────────────────────────────────────────────────────────────

def build_nav(track):
    """Build navigation bar HTML."""
    parts = []
    for t in TRACKS:
        if t == track:
            parts.append(f"<b>{TRACK_META[t]['emoji']} {TRACK_DISPLAY[t]}</b>")
        else:
            parts.append(f'<a href="../{t}/README.md">{TRACK_META[t]["emoji"]} {TRACK_DISPLAY[t]}</a>')
    return (
        '<div align="center">\n'
        '  <a href="../README.md">🏠 Home</a> · \n'
        '  <a href="../00_The_Architects_Rubric.md">📋 Rubric</a> · \n'
        f'  {" · ".join(parts)}\n'
        '</div>'
    )


def write_scope_file(track, scope_info, questions, output_dir):
    """Write a single scope file with competency sub-sections."""
    filepath = output_dir / f"{scope_info['slug']}.md"
    nav = build_nav(track)

    lines = []
    lines.append(f"# {scope_info['title']}\n")
    lines.append(nav)
    lines.append("\n---\n")
    lines.append(f"*{scope_info['subtitle']}*\n")
    lines.append(f"{scope_info['description']}\n")
    lines.append(f'> **[➕ Add a Flashcard](https://github.com/harvard-edge/cs249r_book/edit/dev/interviews/{track}/{scope_info["slug"]}.md)** (Edit in Browser) — see [README](../README.md#question-format) for the template.\n')
    lines.append("---\n")

    if not scope_info["sections"]:
        # Visual debugging — just dump all questions sorted by level
        questions.sort(key=lambda q: level_sort_key(q["level"]))
        for q in questions:
            lines.append(q["body"])
            lines.append("")
    else:
        # Group by section, then by level within each section
        by_section = OrderedDict()
        section_names = [s[0] for s in scope_info["sections"]]
        for name in section_names:
            by_section[name] = []
        by_section["Additional Topics"] = []

        for q in questions:
            label = get_section_label(q, scope_info["sections"])
            # Match to closest defined section
            if label in by_section:
                by_section[label].append(q)
            else:
                # Try partial match
                placed = False
                for sec_name in section_names:
                    if any(word in label.lower() for word in sec_name.lower().split()):
                        by_section[sec_name].append(q)
                        placed = True
                        break
                if not placed:
                    by_section["Additional Topics"].append(q)

        # Write sections
        first_section = True
        for (sec_title, sec_emoji) in scope_info["sections"]:
            qs = by_section.get(sec_title, [])
            if not qs:
                continue

            if not first_section:
                lines.append("\n---\n")
            first_section = False

            lines.append(f"\n### {sec_emoji} {sec_title}\n")

            # Group by level within section
            by_level = defaultdict(list)
            for q in qs:
                by_level[q["level"]].append(q)

            for level in ["L3", "L4", "L5", "L6+"]:
                if level not in by_level:
                    continue
                lines.append(f"\n{LEVEL_HEADERS[level]}\n")
                for q in by_level[level]:
                    lines.append(q["body"])
                    lines.append("")

        # Additional topics (overflow)
        overflow = by_section.get("Additional Topics", [])
        if overflow:
            lines.append("\n---\n")
            lines.append("\n### 📎 Additional Topics\n")
            by_level = defaultdict(list)
            for q in overflow:
                by_level[q["level"]].append(q)
            for level in ["L3", "L4", "L5", "L6+"]:
                if level not in by_level:
                    continue
                lines.append(f"\n{LEVEL_HEADERS[level]}\n")
                for q in by_level[level]:
                    lines.append(q["body"])
                    lines.append("")

    with open(filepath, "w") as f:
        f.write("\n".join(lines))

    return len(questions)


def write_track_readme(track, scope_data, total_questions, output_dir):
    """Write the track README with scope-based index."""
    meta = TRACK_META[track]
    nav = build_nav(track)

    lines = []
    lines.append(f"# {meta['emoji']} {meta['name']}\n")
    lines.append(nav)
    lines.append("\n---\n")
    lines.append(f"{meta['description']}\n")
    lines.append("### The Constraint Regime\n")
    lines.append(meta["constraint_table"])
    lines.append("\n")
    lines.append("### The Learning Journey\n")
    lines.append("Each file represents a **system scope** — the system you're reasoning about. Within each file, questions are organized by competency topic and mastery level.\n")

    # Scope table
    lines.append("<table>")
    lines.append("  <thead>")
    lines.append("    <tr>")
    lines.append('      <th width="5%">#</th>')
    lines.append('      <th width="30%">Scope</th>')
    lines.append('      <th width="35%">What you\'re studying</th>')
    lines.append('      <th width="10%">L3</th>')
    lines.append('      <th width="10%">L4</th>')
    lines.append('      <th width="10%">L5</th>')
    lines.append('      <th width="10%">L6+</th>')
    lines.append('      <th width="10%">Total</th>')
    lines.append("    </tr>")
    lines.append("  </thead>")
    lines.append("  <tbody>")

    for slug, (title, subtitle, level_counts, count) in scope_data.items():
        num = slug.split("_")[0]
        lines.append("    <tr>")
        lines.append(f'      <td><b>{num}</b></td>')
        lines.append(f'      <td><b><a href="{slug}.md">{title}</a></b></td>')
        lines.append(f'      <td><i>{subtitle}</i></td>')
        lines.append(f'      <td>{level_counts.get("L3", "—")}</td>')
        lines.append(f'      <td>{level_counts.get("L4", "—")}</td>')
        lines.append(f'      <td>{level_counts.get("L5", "—")}</td>')
        lines.append(f'      <td>{level_counts.get("L6+", "—")}</td>')
        lines.append(f"      <td><b>{count}</b></td>")
        lines.append("    </tr>")

    lines.append("    <tr>")
    lines.append(f'      <td></td><td><b>Total</b></td><td></td><td></td><td></td><td></td><td></td><td><b>{total_questions}</b></td>')
    lines.append("    </tr>")
    lines.append("  </tbody>")
    lines.append("</table>\n")

    lines.append(f"### Who This Track Is For\n")
    lines.append(f"{meta['audience']}\n")

    with open(output_dir / "README.md", "w") as f:
        f.write("\n".join(lines))


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("  Interview Playbook Reorganizer v2: Scope-First")
    print("=" * 60)

    for track in TRACKS:
        track_dir = BASE_DIR / track
        legacy_dir = track_dir / "_legacy"
        scopes = TRACK_SCOPES[track]

        print(f"\n{'─' * 60}")
        print(f"  Processing: {track.upper()}")
        print(f"{'─' * 60}")

        # Read from _legacy (originals)
        all_questions = []
        visual_questions = []

        source_dir = legacy_dir if legacy_dir.exists() else track_dir
        for fname in sorted(os.listdir(source_dir)):
            if not fname.endswith(".md") or fname == "README.md":
                continue
            filepath = source_dir / fname
            questions = extract_questions(filepath)
            if "visual" in fname.lower() or "debugging" in fname.lower():
                visual_questions.extend(questions)
                print(f"  📸 {fname}: {len(questions)} visual questions")
            else:
                all_questions.extend(questions)
                print(f"  📄 {fname}: {len(questions)} questions")

        total = len(all_questions) + len(visual_questions)
        print(f"\n  Total: {total}")

        # Classify into scopes
        by_scope = defaultdict(list)
        for q in all_questions:
            scope = classify_scope(q, track)
            by_scope[scope].append(q)

        # Add visual debugging
        visual_slug = [s["slug"] for s in scopes if "visual" in s["slug"]]
        if visual_slug and visual_questions:
            by_scope[visual_slug[0]] = visual_questions

        # Print distribution
        print(f"\n  Distribution by scope:")
        scope_data = OrderedDict()
        total_output = 0
        for scope_info in scopes:
            slug = scope_info["slug"]
            qs = by_scope.get(slug, [])
            if not qs:
                continue
            level_counts = defaultdict(int)
            for q in qs:
                level_counts[q["level"]] += 1
            scope_data[slug] = (scope_info["title"], scope_info["subtitle"], dict(level_counts), len(qs))
            total_output += len(qs)
            level_str = " | ".join(f"{lvl}: {level_counts.get(lvl, 0)}" for lvl in ["L3", "L4", "L5", "L6+"])
            print(f"    {scope_info['title']:40s} {len(qs):4d}  ({level_str})")

        print(f"\n  ✅ Input: {total} | Output: {total_output} | {'MATCH' if total == total_output else 'MISMATCH ⚠️'}")

        if args.dry_run:
            continue

        # Clean old generated files (not _legacy)
        for fname in sorted(os.listdir(track_dir)):
            fpath = track_dir / fname
            if fname.endswith(".md") and fname != "README.md" and not fname.startswith("_") and fpath.is_file():
                fpath.unlink()
                print(f"  🗑️  Removed old {fname}")

        # Write new scope files
        for scope_info in scopes:
            slug = scope_info["slug"]
            qs = by_scope.get(slug, [])
            if not qs:
                continue
            count = write_scope_file(track, scope_info, qs, track_dir)
            print(f"  ✍️  Wrote {slug}.md ({count} questions)")

        # Write README
        write_track_readme(track, scope_data, total_output, track_dir)
        print(f"  ✍️  Wrote README.md")

    print(f"\n{'=' * 60}")
    print("  Done!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

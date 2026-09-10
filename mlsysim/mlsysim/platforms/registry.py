"""Deployment paradigm envelopes (Cloud, Edge, Mobile, TinyML)."""

from ..core.units import (
    GB,
    GiB,
    KiB,
    MB,
    PFLOP,
    TB,
    TFLOPs,
    TOPS,
    milliwatt,
    second,
    ureg,
)
from ..core.registry import Registry
from ..core.types import Metadata
from ..core import provenance_catalog as pc
from .types import PlatformEnvelope

# Latency envelopes (ms, prose ranges for tables)
CLOUD_LATENCY_RANGE_MS = "100-500"
EDGE_LATENCY_RANGE_MS = "10-100"
MOBILE_LATENCY_RANGE_MS = "5-50"
TINY_LATENCY_RANGE_MS = "1-10"

# Resource envelopes (prose ranges for tables)
MOBILE_RAM_RANGE_GB = "8-16"
MOBILE_STORAGE_RANGE = "128 GB-1 TB"
MOBILE_TDP_RANGE_W = "3-5"

# Reference capacities (quantities for napkin math)
CLOUD_MEM_GIB = 100 * GiB
MOBILE_MEM_GIB = 8 * GiB
TINY_MEM_KIB = 512 * KiB
SMARTPHONE_RAM_GIB = 8 * GiB
MCU_RAM_KIB = 512 * KiB


class Platforms(Registry):
    """Deployment paradigms with latency, memory, and power envelopes."""

    Cloud = PlatformEnvelope(
        name="Cloud",
        ram=CLOUD_MEM_GIB,
        storage=10 * TB,
        typical_latency_budget=200 * ureg.ms,
        latency_range_ms=CLOUD_LATENCY_RANGE_MS,
        ram_range=f"{CLOUD_MEM_GIB.to('GiB').magnitude:.0f}+ GiB",
        compute_threshold=1000 * (TFLOPs / second),
        bandwidth_threshold=1000 * (GB / second),
        metadata=Metadata(provenance=pc.DEPLOYMENT_ENVELOPES),
    )
    Edge = PlatformEnvelope(
        name="Edge",
        ram=32 * GiB,
        storage=1 * TB,
        typical_latency_budget=50 * ureg.ms,
        latency_range_ms=EDGE_LATENCY_RANGE_MS,
        compute_threshold=1 * (PFLOP / second),
        bandwidth_threshold=270 * (GB / second),
        metadata=Metadata(provenance=pc.DEPLOYMENT_ENVELOPES),
    )
    Mobile = PlatformEnvelope(
        name="Mobile",
        ram=SMARTPHONE_RAM_GIB,
        storage=256 * GB,
        typical_latency_budget=30 * ureg.ms,
        latency_range_ms=MOBILE_LATENCY_RANGE_MS,
        ram_range=MOBILE_RAM_RANGE_GB,
        storage_range=MOBILE_STORAGE_RANGE,
        tdp_range_w=MOBILE_TDP_RANGE_W,
        metadata=Metadata(provenance=pc.DEPLOYMENT_ENVELOPES),
    )
    Tiny = PlatformEnvelope(
        name="TinyML",
        ram=MCU_RAM_KIB,
        storage=4 * ureg.MiB,
        typical_latency_budget=100 * ureg.ms,
        latency_range_ms=TINY_LATENCY_RANGE_MS,
        ram_range=f"{TINY_MEM_KIB.to('KiB').magnitude:.0f} KiB",
        compute_threshold=1 * TOPS,
        power_threshold=1 * milliwatt,
        metadata=Metadata(provenance=pc.DEPLOYMENT_ENVELOPES),
    )

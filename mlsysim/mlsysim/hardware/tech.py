"""Hardware technology-class reference facts (exposed as ``Hardware.Tech``).

Technology-CLASS properties — access latency, per-operation/per-byte energy, and generic
component bandwidth — that are ~constant across parts of a generation. This is the
counterpart to the per-instance specs on the Hardware.Cloud/Edge/Mobile/Tiny nodes
(capacity, bandwidth, TDP, price), which genuinely vary part-to-part.
See .claude/rules/mlsysim.md -> Canonical organization (the instance-vs-tech-class split).

The tier *data* lives as YAML under ``hardware/data/tech/<category>.yaml`` (loaded +
validated against the tier schemas below); the tier *types*, the ``Tech`` namespace, and
the instance→tech-class fallback resolver are code. Energy figures are Horowitz (2014),
45 nm — process-stamped in the names and provenance so a future process node can coexist
rather than overwrite the teaching anchor (invariant #3).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from ..core.types import Quantity, Metadata
from ..core.registry import Registry
from ..core.loader import load_collection


class MemoryTier(BaseModel):
    """On-chip / off-chip memory technology tier: access latency + access energy."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str
    latency: Optional[Quantity] = None            # access latency
    energy_per_access: Optional[Quantity] = None  # pJ per access
    energy_per_byte: Optional[Quantity] = None    # pJ per byte
    metadata: Metadata = Field(default_factory=Metadata)


class StorageTier(BaseModel):
    """Generic storage/memory bandwidth tier (sequential bandwidth + optional latency)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str
    bandwidth: Quantity
    latency: Optional[Quantity] = None
    metadata: Metadata = Field(default_factory=Metadata)


class InterconnectTier(BaseModel):
    """Interconnect technology generation: link access latency (technology-class).

    Per-instance link *bandwidth* lives on each node's IOInterconnect (it varies
    part-to-part); the access *latency* floor is a property of the interconnect
    generation, ~constant across parts, so it lives here."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str
    latency: Optional[Quantity] = None
    metadata: Metadata = Field(default_factory=Metadata)


class OpEnergy(BaseModel):
    """Per-operation energy for an arithmetic op (technology/process-class)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str
    energy: Quantity
    metadata: Metadata = Field(default_factory=Metadata)


_DATA = Path(__file__).parent / "data" / "tech"

Memory = load_collection(
    _DATA / "memory.yaml", MemoryTier, name="Memory",
    doc="On-chip memory hierarchy tiers (access latency + 45 nm access energy).",
)
Storage = load_collection(
    _DATA / "storage.yaml", StorageTier, name="Storage",
    doc="Generic storage / off-chip memory bandwidth tiers.",
)
Op = load_collection(
    _DATA / "op.yaml", OpEnergy, name="Op",
    doc="Per-operation arithmetic energy (Horowitz 2014, 45 nm).",
)
Interconnect = load_collection(
    _DATA / "interconnect.yaml", InterconnectTier, name="Interconnect",
    doc="Interconnect technology generations (link access-latency floor).",
)


class Tech(Registry):
    """Technology-class reference facts: latency / energy / generic bandwidth."""

    Memory = Memory
    Storage = Storage
    Op = Op
    Interconnect = Interconnect


def resolve_latency(instance_latency, tech_default):
    """Invariant #1 fallback: use the instance's own measured value when set, else the
    technology-class default. e.g. ``resolve_latency(node.memory.latency, Tech.Memory.HBM3.latency)``.
    Lets a specific chip override a class fact without forcing per-instance values everywhere."""
    return instance_latency if instance_latency is not None else tech_default

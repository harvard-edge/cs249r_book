"""Hardware registry — accelerator/device instances.

Per-instance specs (capacity, bandwidth, TDP, price, counts) live as YAML data
under ``hardware/data/<tier>/<Chip>.yaml`` and are loaded + validated against the
``HardwareNode`` schema at import (see ``core/loader.py`` and
``.claude/rules/mlsysim.md`` → *Canonical organization* / *Storage format*).
Technology-class facts that are ~constant across a generation (access latency,
per-op/byte energy, generic bandwidth) live in ``hardware/tech.py`` and are
exposed as ``Hardware.Tech``; instances reference them via ``@tech:`` markers in
their YAML so a value lives in exactly one place.
"""
from pathlib import Path

from ..core.registry import Registry
from ..core.loader import load_registry
from .types import HardwareNode
from .tech import Tech as _Tech  # technology-class facts (NVLink latency, etc.)

_DATA = Path(__file__).parent / "data"

CloudHardware = load_registry(
    _DATA / "cloud", HardwareNode, name="CloudHardware",
    doc="Datacenter-scale accelerators (Volume II).", tech_root=_Tech,
)
WorkstationHardware = load_registry(
    _DATA / "workstation", HardwareNode, name="WorkstationHardware",
    doc="Personal computing systems used for local development.", tech_root=_Tech,
)
MobileHardware = load_registry(
    _DATA / "mobile", HardwareNode, name="MobileHardware",
    doc="Smartphone and handheld devices (Volume I).", tech_root=_Tech,
)
EdgeHardware = load_registry(
    _DATA / "edge", HardwareNode, name="EdgeHardware",
    doc="Robotics and Industrial Edge (Volume I).", tech_root=_Tech,
)
TinyHardware = load_registry(
    _DATA / "tiny", HardwareNode, name="TinyHardware",
    doc="Microcontrollers and sub-watt devices.", tech_root=_Tech,
)


class Hardware(Registry):
    """Registry namespace for Hardware."""
    Cloud = CloudHardware
    Workstation = WorkstationHardware
    Mobile = MobileHardware
    Edge = EdgeHardware
    Tiny = TinyHardware
    Tech = _Tech  # technology-class facts (latency / energy / generic bandwidth)

"""Lineage gates for numbers that appear in assumption appendices."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ..core.provenance import Provenance, Sourced

_REPO_ROOT = Path(__file__).resolve().parents[3]
APPENDIX_ASSUMPTIONS_QMD = (
    _REPO_ROOT / "book/quarto/contents/vol1/backmatter/appendix_assumptions.qmd",
    _REPO_ROOT / "book/quarto/contents/vol2/backmatter/appendix_assumptions.qmd",
)

# Legacy pattern — should be empty after migration.
_DEFAULTS_REF = re.compile(r"\bdefaults\.([A-Z][A-Z0-9_]+)\b")
_PRICING_REF = re.compile(
    r"\bInfrastructure\.Pricing\.(Cloud|Storage|Labeling|Fleet|Capital)\.(\w+)(?:\.rate)?\b"
)
_RELIABILITY_REF = re.compile(
    r"\bSystems\.Reliability\.(\w+)(?:\.mttf_hours|\.Recovery\.(\w+))?\b"
)
_LITERATURE_REF = re.compile(
    r"\bLiterature\.(Training|Scaling|Overheads|Chinchilla|Communication)\.(\w+)\b"
)


def _appendix_text() -> str:
    """Reads the content of an appendix .qmd file with caching to prevent redundant disk I/O."""
    parts: list[str] = []
    for path in APPENDIX_ASSUMPTIONS_QMD:
        if path.is_file():
            parts.append(path.read_text(encoding="utf-8"))
    return "\n".join(parts)


def provenance_for_value(value: Any) -> Provenance | None:
    if isinstance(value, Sourced):
        return value.provenance
    if hasattr(value, "mttf_hours") and isinstance(value.mttf_hours, Sourced):
        return value.mttf_hours.provenance
    recovery_fields = ("heartbeat_timeout_s", "reschedule_time_s", "checkpoint_write_bw_gbs")
    for field in recovery_fields:
        if hasattr(value, field):
            v = getattr(value, field)
            if isinstance(v, Sourced):
                return v.provenance
    meta = getattr(value, "metadata", None)
    if meta is not None:
        return getattr(meta, "provenance", None)
    return None


def audit_appendix_defaults() -> list[str]:
    """Reject any remaining ``defaults.*`` references in assumption appendices."""
    issues: list[str] = []
    for symbol in sorted(set(_DEFAULTS_REF.findall(_appendix_text()))):
        issues.append(f"defaults.{symbol}: remove — use Systems/Literature/Infrastructure registries")
    return issues


def audit_appendix_pricing() -> list[str]:
    from ..infrastructure.registry import Infrastructure

    issues: list[str] = []
    section_map = {
        "Cloud": Infrastructure.Pricing.Cloud,
        "Storage": Infrastructure.Pricing.Storage,
        "Labeling": Infrastructure.Pricing.Labeling,
        "Fleet": Infrastructure.Pricing.Fleet,
        "Capital": Infrastructure.Pricing.Capital,
    }
    for section, entry in sorted(set(_PRICING_REF.findall(_appendix_text()))):
        reg = section_map.get(section)
        path = f"Infrastructure.Pricing.{section}.{entry}"
        if reg is None or not hasattr(reg, entry):
            issues.append(f"{path}: referenced in appendix but undefined")
            continue
        point = getattr(reg, entry)
        if provenance_for_value(point) is None:
            issues.append(f"{path}: used in appendix without provenance")
    return issues


def audit_appendix_reliability() -> list[str]:
    from ..systems.reliability import Reliability

    issues: list[str] = []
    text = _appendix_text()
    if "Systems.Reliability." not in text:
        return issues
    for comp_name in ("Gpu", "Nic", "Psu", "PcieSwitch", "Cable", "TorSwitch", "Hbm"):
        if f"Systems.Reliability.{comp_name}" in text:
            comp = getattr(Reliability, comp_name)
            if provenance_for_value(comp) is None:
                issues.append(f"Systems.Reliability.{comp_name}: missing provenance")
    if "Systems.Reliability.Recovery" in text:
        if provenance_for_value(Reliability.Recovery) is None:
            issues.append("Systems.Reliability.Recovery: missing provenance")
    return issues


def audit_appendix_literature() -> list[str]:
    from ..literature.registry import Literature

    issues: list[str] = []
    section_map = {
        "Training": Literature.Training,
        "Scaling": Literature.Scaling,
        "Overheads": Literature.Overheads,
        "Chinchilla": Literature.Chinchilla,
        "Communication": Literature.Communication,
    }
    for section, attr in sorted(set(_LITERATURE_REF.findall(_appendix_text()))):
        reg = section_map.get(section)
        path = f"Literature.{section}.{attr}"
        if reg is None or not hasattr(reg, attr):
            issues.append(f"{path}: referenced in appendix but undefined")
            continue
        val = getattr(reg, attr)
        if isinstance(val, Sourced) and val.provenance is None:
            issues.append(f"{path}: missing provenance")
    return issues

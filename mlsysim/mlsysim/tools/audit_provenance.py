#!/usr/bin/env python3
"""Report missing or weak provenance on registry entries."""

from __future__ import annotations

import argparse
import sys
from typing import Any, Iterable

from mlsysim.tools.appendix_lineage import (
    audit_appendix_defaults,
    audit_appendix_literature,
    audit_appendix_pricing,
    audit_appendix_reliability,
)
from mlsysim.core.provenance import Provenance, ProvenanceKind, Sourced
from mlsysim.hardware.registry import (
    CloudHardware,
    EdgeHardware,
    MobileHardware,
    TinyHardware,
    WorkstationHardware,
)
from mlsysim.infrastructure.registry import Grids
from mlsysim.infrastructure.pricing import Cloud, Storage, Labeling, Fleet, Capital, OnPremises
from mlsysim.infrastructure.capacity import Capacity
from mlsysim.literature.registry import Training, Scaling, Overheads, Chinchilla, Communication
from mlsysim.models.registry import (
    GenerativeVisionModels,
    LanguageModels,
    RecommendationModels,
    StateSpaceModels,
    TinyModels,
    VisionModels,
)
def _registry_nodes(registry_cls: type) -> Iterable[Any]:
    """Yields all Sourced AST nodes found in the target registry file."""
    if not hasattr(registry_cls, "list"):
        return []
    return registry_cls.list()


def _validate_provenance_record(path: str, prov: Provenance | None) -> list[str]:
    """Validates that a Provenance record meets the required traceability constraints."""
    if prov is None:
        return [f"{path}: missing provenance"]
    issues: list[str] = []
    if not prov.ref.strip():
        issues.append(f"{path}: empty provenance.ref")
    if prov.kind == ProvenanceKind.DATASHEET and not prov.url:
        issues.append(f"{path}: datasheet without url")
    if prov.kind == ProvenanceKind.ESTIMATE and not prov.notes:
        issues.append(f"{path}: estimate without notes")
    if prov.kind == ProvenanceKind.DERIVED and not prov.notes:
        issues.append(f"{path}: derived without notes")
    if prov.verified and len(prov.verified) != 10:
        issues.append(f"{path}: verified date must be YYYY-MM-DD")
    return issues


def _check_node(path: str, node: Any) -> list[str]:
    """Inspects an AST node to verify its provenance lineage."""
    meta = getattr(node, "metadata", None)
    if meta is not None:
        return _validate_provenance_record(path, getattr(meta, "provenance", None))
    if isinstance(node, Sourced):
        return _validate_provenance_record(path, node.provenance)
    if hasattr(node, "mttf_hours"):
        return _validate_provenance_record(path, getattr(node.mttf_hours, "provenance", None))
    if hasattr(node, "rate"):
        return _check_node(path, node)
    return []


def audit_registries(*, scope_cloud: bool = False) -> list[str]:
    issues: list[str] = []
    groups = (
        [("Hardware.Cloud", CloudHardware)]
        if scope_cloud
        else [
            ("Hardware.Cloud", CloudHardware),
            ("Hardware.Workstation", WorkstationHardware),
            ("Hardware.Mobile", MobileHardware),
            ("Hardware.Edge", EdgeHardware),
            ("Hardware.Tiny", TinyHardware),
            ("Models.LanguageModels", LanguageModels),
            ("Models.VisionModels", VisionModels),
            ("Models.TinyModels", TinyModels),
            ("Models.RecommendationModels", RecommendationModels),
            ("Models.StateSpaceModels", StateSpaceModels),
            ("Models.GenerativeVisionModels", GenerativeVisionModels),
        ]
    )
    for prefix, reg in groups:
        for node in _registry_nodes(reg):
            name = getattr(node, "name", type(node).__name__)
            issues.extend(_check_node(f"{prefix}.{name}", node))
    return issues


def audit_infra_grids() -> list[str]:
    issues: list[str] = []
    for grid in _registry_nodes(Grids):
        name = getattr(grid, "name", type(grid).__name__)
        issues.extend(_check_node(f"Infrastructure.Grids.{name}", grid))
    return issues


def audit_infra_pricing() -> list[str]:
    issues: list[str] = []
    for prefix, reg in (
        ("Infrastructure.Pricing.Cloud", Cloud),
        ("Infrastructure.Pricing.Storage", Storage),
        ("Infrastructure.Pricing.Labeling", Labeling),
        ("Infrastructure.Pricing.Fleet", Fleet),
        ("Infrastructure.Pricing.Capital", Capital),
        ("Infrastructure.Pricing.OnPremises", OnPremises),
    ):
        for point in _registry_nodes(reg):
            name = getattr(point, "name", type(point).__name__)
            issues.extend(_check_node(f"{prefix}.{name}", point))
    return issues


def audit_infra_capacity() -> list[str]:
    issues: list[str] = []
    for val in _registry_nodes(Capacity):
        if isinstance(val, Sourced):
            issues.extend(_validate_provenance_record("Infrastructure.Capacity", val.provenance))
    return issues


def audit_literature_sourced() -> list[str]:
    issues: list[str] = []
    for prefix, reg in (
        ("Literature.Training", Training),
        ("Literature.Scaling", Scaling),
        ("Literature.Overheads", Overheads),
        ("Literature.Chinchilla", Chinchilla),
        ("Literature.Communication", Communication),
    ):
        for item in _registry_nodes(reg):
            if isinstance(item, Sourced):
                issues.extend(_validate_provenance_record(f"{prefix}", item.provenance))
    return issues


def audit_calibration_sourced() -> list[str]:
    issues: list[str] = []
    from mlsysim.engine import calibration as cal

    for name in dir(cal):
        if name.startswith("_"):
            continue
        val = getattr(cal, name)
        if isinstance(val, Sourced):
            issues.extend(
                _validate_provenance_record(f"engine.calibration.{name}", val.provenance)
            )
    return issues


def audit_systems_reliability() -> list[str]:
    from mlsysim.systems.reliability import Reliability

    issues: list[str] = []
    for comp in _registry_nodes(Reliability):
        if hasattr(comp, "name"):
            issues.extend(_check_node(f"Systems.Reliability.{comp.name}", comp))
    recovery = Reliability.Recovery
    for field in ("heartbeat_timeout_s", "reschedule_time_s", "checkpoint_write_bw_gbs"):
        val = getattr(recovery, field)
        if isinstance(val, Sourced):
            issues.extend(
                _validate_provenance_record(f"Systems.Reliability.Recovery.{field}", val.provenance)
            )
    return issues


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scope",
        choices=("cloud", "all", "textbook"),
        default="textbook",
        help="What to scan",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit 1 when any issue is found",
    )
    args = parser.parse_args(argv)

    issues: list[str] = []
    if args.scope == "cloud":
        issues.extend(audit_registries(scope_cloud=True))
    if args.scope in ("all", "textbook"):
        issues.extend(audit_registries(scope_cloud=False))
        issues.extend(audit_infra_grids())
        issues.extend(audit_infra_pricing())
        issues.extend(audit_infra_capacity())
        issues.extend(audit_literature_sourced())
        issues.extend(audit_systems_reliability())
        issues.extend(audit_calibration_sourced())
    if args.scope == "textbook":
        issues.extend(audit_appendix_defaults())
        issues.extend(audit_appendix_pricing())
        issues.extend(audit_appendix_reliability())
        issues.extend(audit_appendix_literature())

    if issues:
        print(f"Provenance audit ({args.scope}): {len(issues)} issue(s)")
        for line in sorted(issues):
            print(f"  - {line}")
        return 1 if args.strict else 0

    print(f"Provenance audit OK (scope={args.scope}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

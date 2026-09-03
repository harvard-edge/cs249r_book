#!/usr/bin/env python3
"""Report missing or weak provenance on registry entries."""

from __future__ import annotations

import argparse
import sys
from datetime import date
from typing import Any, Iterable

from mlsysim.core.provenance import Provenance, ProvenanceKind, Sourced
from mlsysim.datasets.registry import Datasets
from mlsysim.hardware.registry import (
    CloudHardware,
    EdgeHardware,
    MobileHardware,
    TinyHardware,
    WorkstationHardware,
)
from mlsysim.hardware.tech import Interconnect, Memory, Op, Storage as TechStorage
from mlsysim.infrastructure.registry import Datacenters, FacilityCooling, Grids, Racks
from mlsysim.infrastructure.pricing import Cloud, Storage, Labeling, Fleet, Capital, OnPremises
from mlsysim.infrastructure.capacity import Capacity
from mlsysim.literature.registry import (
    BatchSize,
    Benchmarks,
    Chinchilla,
    Communication,
    Training,
)
from mlsysim.models.registry import (
    GenerativeVisionModels,
    LanguageModels,
    RecommendationModels,
    StateSpaceModels,
    TinyModels,
    VisionModels,
)
from mlsysim.ops.monitoring import Monitoring
from mlsysim.ops.runtime import MemoryProtection, RuntimeOverheads
from mlsysim.ops.training import TrainingRunOverheads
from mlsysim.platforms.registry import Platforms
from mlsysim.reference_stats.registry import ReferenceStats
from mlsysim.systems.registry import (
    Clusters,
    Fabrics,
    NetworkEnergy,
    Nodes,
    Pods,
    Racks as SystemRacks,
    Storage as SystemStorage,
    SwitchFabric,
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
    if not prov.verified:
        issues.append(f"{path}: missing verified date")
    else:
        try:
            date.fromisoformat(prov.verified)
        except ValueError:
            issues.append(f"{path}: verified date must be YYYY-MM-DD")
    if prov.kind in {
        ProvenanceKind.DATASHEET,
        ProvenanceKind.LITERATURE,
        ProvenanceKind.INDUSTRY_REPORT,
    } and not prov.url:
        issues.append(f"{path}: {prov.kind.value} without url")
    if prov.kind in {ProvenanceKind.ESTIMATE, ProvenanceKind.DERIVED} and not prov.notes:
        issues.append(f"{path}: {prov.kind.value} without notes")
    return issues


def _check_node(path: str, node: Any) -> list[str]:
    """Inspects a runtime registry object to verify its provenance lineage."""
    meta = getattr(node, "metadata", None)
    if meta is not None:
        return _validate_provenance_record(path, getattr(meta, "provenance", None))
    if isinstance(node, Sourced):
        return _validate_provenance_record(path, node.provenance)
    if hasattr(node, "provenance"):
        return _validate_provenance_record(path, getattr(node, "provenance", None))
    if hasattr(node, "mttf_hours"):
        return _validate_provenance_record(path, getattr(node.mttf_hours, "provenance", None))
    if hasattr(node, "rate"):
        # Recurse into the rate VALUE (was `node`, an infinite loop — audit fix 2026-06-06).
        return _check_node(path, node.rate)
    # A value that reaches here carries no provenance hook at all. Report it
    # instead of silently passing (audit fix 2026-06-06: bare floats/Quantities
    # previously returned [] and made the "every value is sourced" claim
    # unenforced).
    return [f"{path}: no provenance attached ({type(node).__name__})"]


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


def audit_datasets() -> list[str]:
    issues: list[str] = []
    for dataset in _registry_nodes(Datasets):
        name = getattr(dataset, "name", type(dataset).__name__)
        issues.extend(_check_node(f"Datasets.{name}", dataset))
    return issues


def audit_platforms() -> list[str]:
    issues: list[str] = []
    for platform in _registry_nodes(Platforms):
        name = getattr(platform, "name", type(platform).__name__)
        issues.extend(_check_node(f"Platforms.{name}", platform))
    return issues


def audit_hardware_tech() -> list[str]:
    issues: list[str] = []
    for prefix, reg in (
        ("Hardware.Tech.Memory", Memory),
        ("Hardware.Tech.Storage", TechStorage),
        ("Hardware.Tech.Op", Op),
        ("Hardware.Tech.Interconnect", Interconnect),
    ):
        for node in _registry_nodes(reg):
            name = getattr(node, "name", type(node).__name__)
            issues.extend(_check_node(f"{prefix}.{name}", node))
    return issues


def audit_systems_topology() -> list[str]:
    issues: list[str] = []
    for prefix, reg in (
        ("Systems.Nodes", Nodes),
        ("Systems.Fabrics", Fabrics),
        ("Systems.Clusters", Clusters),
        ("Systems.Pods", Pods),
        ("Systems.Racks", SystemRacks),
        ("Systems.Storage", SystemStorage),
    ):
        for node in _registry_nodes(reg):
            name = getattr(node, "name", type(node).__name__)
            issues.extend(_check_node(f"{prefix}.{name}", node))
    return issues


def audit_systems_reference_values() -> list[str]:
    issues: list[str] = []
    for prefix, reg in (
        ("Systems.SwitchFabric", SwitchFabric),
        ("Systems.NetworkEnergy", NetworkEnergy),
    ):
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


def audit_infra_facilities() -> list[str]:
    issues: list[str] = []
    for prefix, reg in (
        ("Infrastructure.Datacenters", Datacenters),
        ("Infrastructure.Racks", Racks),
        ("Infrastructure.FacilityCooling", FacilityCooling),
    ):
        for node in _registry_nodes(reg):
            name = getattr(node, "name", type(node).__name__)
            issues.extend(_check_node(f"{prefix}.{name}", node))
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
        ("Literature.Benchmarks", Benchmarks),
        ("Literature.Chinchilla", Chinchilla),
        ("Literature.Communication", Communication),
        ("Literature.BatchSize", BatchSize),
    ):
        for item in _registry_nodes(reg):
            if isinstance(item, Sourced):
                issues.extend(_validate_provenance_record(f"{prefix}", item.provenance))
    return issues


def audit_reference_stats() -> list[str]:
    issues: list[str] = []
    for item in _registry_nodes(ReferenceStats):
        name = getattr(item, "name", type(item).__name__)
        issues.extend(_check_node(f"ReferenceStats.{name}", item))
    return issues


def audit_ops_sourced() -> list[str]:
    issues: list[str] = []
    for prefix, reg in (
        ("Ops.Monitoring", Monitoring),
        ("Ops.RuntimeOverheads", RuntimeOverheads),
        ("Ops.MemoryProtection", MemoryProtection),
        ("Ops.TrainingRunOverheads", TrainingRunOverheads),
    ):
        for item in _registry_nodes(reg):
            name = getattr(item, "name", type(item).__name__)
            issues.extend(_check_node(f"{prefix}.{name}", item))
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
    for field in (
        "heartbeat_timeout_s",
        "reschedule_time_s",
        "detection_time_s",
        "restart_time_s",
        "warmup_time_s",
        "checkpoint_write_bw_gbs",
    ):
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
        choices=("cloud", "all"),
        default="all",
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
    if args.scope == "all":
        issues.extend(audit_registries(scope_cloud=False))
        issues.extend(audit_datasets())
        issues.extend(audit_platforms())
        issues.extend(audit_hardware_tech())
        issues.extend(audit_systems_topology())
        issues.extend(audit_systems_reference_values())
        issues.extend(audit_infra_grids())
        issues.extend(audit_infra_facilities())
        issues.extend(audit_infra_pricing())
        issues.extend(audit_infra_capacity())
        issues.extend(audit_literature_sourced())
        issues.extend(audit_reference_stats())
        issues.extend(audit_ops_sourced())
        issues.extend(audit_systems_reliability())
        issues.extend(audit_calibration_sourced())

    if issues:
        print(f"Provenance audit ({args.scope}): {len(issues)} issue(s)")
        for line in sorted(issues):
            print(f"  - {line}")
        return 1 if args.strict else 0

    print(f"Provenance audit OK (scope={args.scope}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

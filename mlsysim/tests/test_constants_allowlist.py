"""CI gate: core/constants.py is a units-only re-export — it must define NO values.

The taxonomy refactor (2026-05) emptied this module: every hardware/model/systems/
literature/scenario figure moved to its category registry, every physical constant to
physics/constants.py, and every byte/bit width + PRECISION_MAP to core/units.py. The
migration ratchet has reached zero, so the former MIGRATION_PENDING patterns are now
hard-forbidden alongside the original spec patterns. The strongest guard is simply that
constants.py defines no module-level assignments at all (it is `from .units import *`).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

CONSTANTS_PATH = Path(__file__).resolve().parents[1] / "mlsysim" / "core" / "constants.py"

# Any value matching these belongs in a registry / physics / units — never constants.py.
# (Folded in the former MIGRATION_PENDING patterns at P8 once the backlog hit zero.)
FORBIDDEN_NAME_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^(H100|A100|V100|B200|H200|MI300X|T4|TPU|JETSON|ESP32|DGX)_"),
    re.compile(r"^NVLINK_"),
    re.compile(r"^PCIE_GEN\d"),
    re.compile(r"^INFINIBAND_"),
    re.compile(r"^(H100|A100|V100|B200|H200|MI300X|T4|TPU).*(FLOPS|TOPS)"),
    re.compile(r"^(RESNET|BERT|MOBILENET|ALEXNET|YOLO|DLRM|GPT|LLAMA|STABLE_DIFFUSION).*(FLOPS|PARAMS|PARAM)"),
    re.compile(r"^GPT\d"),
    re.compile(r"^LLAMA"),
    re.compile(r"^IMAGENET"),
    re.compile(r"^MNIST"),
    re.compile(r"^CIFAR"),
    re.compile(r"^GPU_MTTF"),
    re.compile(r"^CLUSTER_"),
    re.compile(r"^PUE_"),
    re.compile(r"^CLOUD_(EGRESS|ELECTRICITY|GPU_)"),
    re.compile(r"^FLEET_"),
    re.compile(r"^CARBON_"),
    re.compile(r"^STORAGE_COST"),
    re.compile(r"^LABELING_COST"),
    re.compile(r"^TPU_POD_"),
    # --- folded in at P8 (migration complete) ---
    re.compile(r"^LATENCY_"),
    re.compile(r"^ENERGY_"),
    re.compile(r"^(NETWORK_|ETHERNET_|SWITCH_|OPTICS_|FABRIC_|FEC_)"),
    re.compile(r"^(NVME_|SYSTEM_MEMORY_BW|HOST_DRAM_BW|LOCAL_NVME)"),
    re.compile(r"^(MOBILE_|PHONE_|BATTERY_|OBJECT_DETECTOR_)"),
    re.compile(r"^(TRANSFORMER_|SYSTOLIC_)"),
    re.compile(r"^(GMAIL_|GOOGLE_|WAYMO_|ANOMALY_)"),
    re.compile(r"^(VIDEO_|IMAGE_|COLOR_DEPTH)"),
    re.compile(r"^(ML_WORKFLOW_|SYNTHETIC_|LOGIC_WALL_)"),
    re.compile(r"^SPEED_OF_LIGHT"),         # physical constants -> physics/constants.py
    re.compile(r".*_FIBER_KM_S$"),
)


def _defined_names(source: str) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
    return names


def test_constants_is_pure_units_reexport() -> None:
    """The strongest invariant: constants.py defines NO module-level values at all.

    Every value has a category home; this module only does ``from .units import *``.
    """
    source = CONSTANTS_PATH.read_text(encoding="utf-8")
    defined = sorted(_defined_names(source))
    assert not defined, (
        "core/constants.py must define no values — it is a units-only re-export. "
        "Move each to its category home (registry / physics / units):\n  "
        + "\n  ".join(defined)
    )


def test_constants_has_no_forbidden_symbol_names() -> None:
    source = CONSTANTS_PATH.read_text(encoding="utf-8")
    defined = _defined_names(source)
    violations = sorted(
        name for name in defined
        if any(p.search(name) for p in FORBIDDEN_NAME_PATTERNS)
    )
    assert not violations, (
        "constants.py must not define hardware/model/fleet/physics symbols:\n  "
        + "\n  ".join(violations)
    )


def test_constants_does_not_reexport_defaults() -> None:
    source = CONSTANTS_PATH.read_text(encoding="utf-8")
    assert "from .defaults import" not in source
    assert "import defaults" not in source


def test_constants_size_budget() -> None:
    """Guard against re-accumulating deleted constants (it should stay a thin shim)."""
    line_count = len(CONSTANTS_PATH.read_text(encoding="utf-8").splitlines())
    # Budget allows the navigational pointer-comments that document where each
    # former resident moved; the file defines no values (see pure-reexport test).
    assert line_count <= 90, f"constants.py grew to {line_count} lines (budget 90)"

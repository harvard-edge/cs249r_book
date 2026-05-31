"""Vol I/II calculations must stay wired to MLSysIM registries.

The book is allowed to have illustrative scenario constants in a chapter, but
model, hardware, dataset, platform, system, infrastructure, literature, and ops
objects used by rendered calculations should resolve through the package API.
"""

from __future__ import annotations

import re
from pathlib import Path

import mlsysim

REPO_ROOT = Path(__file__).resolve().parents[2]
BOOK_ROOTS = (
    REPO_ROOT / "book" / "quarto" / "contents" / "vol1",
    REPO_ROOT / "book" / "quarto" / "contents" / "vol2",
)

REGISTRY_ROOTS = (
    "Models",
    "Hardware",
    "Systems",
    "Infrastructure",
    "Platforms",
    "Datasets",
    "Literature",
    "Scenarios",
    "Ops",
)
REGISTRY_PATH_RE = re.compile(
    r"\b(" + "|".join(REGISTRY_ROOTS) + r")(?:\.[A-Za-z_]\w*)+"
)

# Quarto examples often call methods after a registry value, e.g.
# ``Hardware.Cloud.H100.memory.capacity.m_as(GB)``.  Resolve the registry object
# and stop before the Pint/Python method suffix.
METHOD_TAILS = frozenset({
    "format_benchmark_table",
    "items",
    "keys",
    "m",
    "m_as",
    "magnitude",
    "to",
    "values",
})

SHORTHAND_SUFFIXES = frozenset({"{", "*"})


def _qmd_files() -> list[Path]:
    files: list[Path] = []
    for root in BOOK_ROOTS:
        files.extend(sorted(root.rglob("*.qmd")))
    return files


def _resolve_registry_ref(ref: str) -> object:
    obj: object = mlsysim
    for part in ref.split("."):
        if part in METHOD_TAILS:
            break
        obj = getattr(obj, part)
    return obj


def test_vol1_vol2_registry_paths_resolve() -> None:
    missing: list[str] = []

    for path in _qmd_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        rel = path.relative_to(REPO_ROOT)

        for match in REGISTRY_PATH_RE.finditer(text):
            # Ignore comment shorthand such as ``Hardware.Foo.{A,B}``; concrete
            # paths beside it are still scanned and resolved.
            if text[match.end():match.end() + 1] in SHORTHAND_SUFFIXES:
                continue

            ref = match.group(0)
            try:
                _resolve_registry_ref(ref)
            except AttributeError as exc:
                missing.append(f"{rel}: {ref} ({exc})")

    assert not missing, (
        "Vol I/II QMD registry references must resolve through mlsysim:\n"
        + "\n".join(missing[:80])
        + (f"\n... and {len(missing) - 80} more" if len(missing) > 80 else "")
    )


def test_book_case_study_model_specs_are_registry_backed() -> None:
    """High-signal textbook model anchors should live under Models.*."""

    lenet = mlsysim.Models.Vision.LeNet1
    mixtral = mlsysim.Models.Language.Mixtral_8x7B

    assert lenet.parameters.m_as("param") == 10_000
    assert mixtral.parameters.m_as("Bparam") == 46.7
    assert mixtral.active_parameters.m_as("Bparam") == 12.9
    assert mixtral.experts == 8
    assert mixtral.active_experts_per_token == 2


def test_scenarios_are_executable_model_system_constraint_bundles() -> None:
    """Concrete scenarios compose existing registries rather than redefining facts."""

    assert not hasattr(mlsysim.Scenarios, "Archetypes")

    assert mlsysim.Scenarios.FrontierTraining.workload is mlsysim.Models.Language.GPT4
    assert mlsysim.Scenarios.FrontierTraining.system is mlsysim.Systems.Clusters.Frontier_8K

    assert mlsysim.Scenarios.AutonomousVehicle.workload is mlsysim.Models.Vision.YOLOv8_Nano
    assert mlsysim.Scenarios.AutonomousVehicle.system is mlsysim.Hardware.Edge.JetsonOrinNX

    assert mlsysim.Scenarios.MobileAssistant.workload is mlsysim.Models.Language.Llama3_8B
    assert mlsysim.Scenarios.MobileAssistant.system is mlsysim.Hardware.Mobile.iPhone15Pro

    assert mlsysim.Scenarios.SmartDoorbell.workload is mlsysim.Models.Tiny.WakeVision
    assert mlsysim.Scenarios.SmartDoorbell.system is mlsysim.Hardware.Tiny.ESP32_S3


def test_reference_stats_are_separate_from_executable_scenarios() -> None:
    """Sourced anchors stay under ReferenceStats, with legacy scenario aliases."""

    assert mlsysim.Scenarios.ReferenceStats is mlsysim.ReferenceStats
    assert mlsysim.Scenarios.Workloads is mlsysim.ReferenceStats.Workloads
    assert mlsysim.Scenarios.MobilePower is mlsysim.ReferenceStats.MobilePower
    assert mlsysim.Scenarios.PhoneBattery is mlsysim.ReferenceStats.PhoneBattery


def test_legacy_application_names_alias_executable_scenarios() -> None:
    """Applications remains an alias so existing book cells are not broken."""

    assert mlsysim.Applications is mlsysim.Scenarios
    assert mlsysim.ScenarioBundles is mlsysim.Scenarios
    assert mlsysim.Applications.Frontier is mlsysim.Scenarios.FrontierTraining
    assert mlsysim.Applications.AutoDrive is mlsysim.Scenarios.AutonomousVehicle
    assert mlsysim.Applications.Mobile is mlsysim.Scenarios.MobileAssistant
    assert mlsysim.Applications.Doorbell is mlsysim.Scenarios.SmartDoorbell

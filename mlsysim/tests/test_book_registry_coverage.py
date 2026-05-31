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


def test_book_workload_archetypes_are_model_registry_backed() -> None:
    """Recurring Vol I lighthouse archetypes must point to Models.* objects."""

    work = mlsysim.Scenarios.Archetypes.Workload

    assert work.ResNet50.workload is mlsysim.Models.Vision.ResNet50
    assert work.GPT2.workload is mlsysim.Models.Language.GPT2
    assert work.DLRM.workload is mlsysim.Models.Recommendation.DLRM
    assert work.MobileNetV2.workload is mlsysim.Models.Vision.MobileNetV2
    assert work.KWS.workload is mlsysim.Models.Tiny.DS_CNN

    assert work.ResNet50.archetype == "Compute Beast"
    assert work.GPT2.archetype == "Bandwidth Hog"
    assert work.DLRM.archetype == "Sparse Scatter"
    assert work.KWS.archetype == "Tiny Constraint"


def test_book_fleet_archetypes_are_model_and_system_registry_backed() -> None:
    """Recurring Vol II A/B/C archetypes must resolve to model and system registries."""

    fleet = mlsysim.Scenarios.Archetypes.Fleet

    assert fleet.ArchetypeA.scale_anchor is mlsysim.Models.Language.GPT4
    assert fleet.ArchetypeA.open_reference is mlsysim.Models.Language.Llama3_70B
    assert fleet.ArchetypeA.system is mlsysim.Systems.Clusters.Frontier_8K
    assert fleet.ArchetypeA.primary_communication == "AllReduce"

    assert fleet.ArchetypeB.scale_anchor is mlsysim.Models.Recommendation.DLRM
    assert fleet.ArchetypeB.system is mlsysim.Systems.Clusters.Production_2K
    assert fleet.ArchetypeB.primary_communication == "AllToAll"

    assert fleet.ArchetypeC.scale_anchor is mlsysim.Models.Vision.MobileNetV2
    assert fleet.ArchetypeC.system is mlsysim.Hardware.Mobile.iPhone15Pro
    assert fleet.ArchetypeC.partitioning_strategy == "Federated Learning"


def test_application_missions_match_executable_scenario_bundles() -> None:
    """Book mission metadata must agree with executable Applications.* scenarios."""

    missions = mlsysim.Scenarios.Archetypes.Missions

    assert mlsysim.Applications.Frontier.workload is missions.FrontierTraining.workload
    assert mlsysim.Applications.Frontier.system is missions.FrontierTraining.system

    assert mlsysim.Applications.AutoDrive.workload is missions.AutonomousPerception.workload
    assert mlsysim.Applications.AutoDrive.system is missions.AutonomousPerception.system

    assert mlsysim.Applications.Mobile.workload is missions.MobileAssistant.workload
    assert mlsysim.Applications.Mobile.system is missions.MobileAssistant.system

    assert mlsysim.Applications.Doorbell.workload is missions.SmartDoorbell.workload
    assert mlsysim.Applications.Doorbell.system is missions.SmartDoorbell.system

    assert mlsysim.Applications.Frontier.workload is mlsysim.Models.Language.GPT4
    assert mlsysim.Applications.AutoDrive.workload is mlsysim.Models.Vision.YOLOv8_Nano
    assert mlsysim.Applications.Mobile.workload is mlsysim.Models.Language.Llama3_8B
    assert mlsysim.Applications.Doorbell.workload is mlsysim.Models.Tiny.WakeVision

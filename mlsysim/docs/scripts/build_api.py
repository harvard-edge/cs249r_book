#!/usr/bin/env python3
"""Regenerate the MLSys·im API reference and fail if the public surface drifts.

Runs as a Quarto ``pre-render`` hook (see ``_quarto.yml``) and can be invoked
directly::

    python scripts/build_api.py          # regenerate api/ and check coverage
    python scripts/build_api.py --check  # coverage check only, no rebuild

Why it exists
-------------
The API pages under ``api/`` used to be committed snapshots that nothing
regenerated, so they drifted from the code (the physics page documented no
functions at all). Now ``api/`` is build output: every full render rebuilds it
from docstrings, and the coverage check below fails the build when a public
symbol has no page. To document a new symbol, add it to the ``quartodoc``
block in ``_quarto.yml`` (physics modules pick up new functions automatically).

The check covers:
  * every function exported by ``mlsysim.physics``
  * every concrete solver exported by ``mlsysim.solvers``
  * every class exported by the registry packages that define ``__all__``
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import os
import re
import subprocess
import sys
from pathlib import Path

DOCS = Path(__file__).resolve().parent.parent
API = DOCS / "api"

# Pages that existed before the API was generated from code. Each now
# redirects to the API index so old links keep resolving.
LEGACY_PAGES = [
    "agents", "core", "datasets", "embodied", "engine", "hardware",
    "hardware.registry", "hardware.registry.Hardware", "hardware.types",
    "hardware.types.HardwareNode", "infrastructure", "literature", "models",
    "models.CNNWorkload", "models.registry", "models.registry.Models",
    "models.TransformerWorkload", "models.types", "models.Workload", "ops",
    "physics", "platforms", "solvers", "systems", "systems.Fleet",
    "systems.Node", "systems.registry", "systems.registry.Systems",
    "systems.types",
]

REGISTRY_PACKAGES = [
    "platforms", "datasets", "literature", "ops", "embodied", "sensors",
    "actuators", "agents",
]

# Exported names that are intentionally not given their own page.
SKIP = {
    "embodied.data_sheet_lookup", "embodied.machine_card",
    "embodied.machine_data_sheet",  # documented in the Embodied Zoo
    "platforms.PlatformEnvelope", "datasets.DatasetProfile",
    "sensors.CameraSensor", "sensors.LiDARSensor", "sensors.IMUSensor",
    "agents.ResearchWorkload", "agents.RemediationWorkload",
    "agents.SupportWorkload", "agents.AgentPlatforms",
    "agents.ReferencePlatforms",
}

INDEX_HEADER = """---
title: "API Reference"
description: "Every public MLSys·im model, registry, solver, and physics formula, generated from the package docstrings."
toc: false
aliases:
{aliases}
---

::: {{.callout-note appearance="simple"}}
This reference is generated from the docstrings in the `mlsysim` package on
every build, so it always matches the installed version. To browse the
contents of a registry (every GPU, model, or fleet), use [the Zoo](../zoo/index.qmd).
:::

"""


def build() -> None:
    subprocess.run(
        [sys.executable, "-m", "quartodoc", "build"], cwd=DOCS, check=True
    )
    index = API / "index.qmd"
    body = index.read_text(encoding="utf-8")
    body = re.sub(r"^# API Reference \{[^}]*\}\n+", "", body)
    aliases = "\n".join(f"  - {name}.html" for name in LEGACY_PAGES)
    index.write_text(INDEX_HEADER.format(aliases=aliases) + body, encoding="utf-8")


def documented_anchors() -> set[str]:
    anchors: set[str] = set()
    for page in API.glob("*.qmd"):
        anchors.update(re.findall(r"\{ #(mlsysim\.[\w.]+) \}", page.read_text(encoding="utf-8")))
    return anchors


def short(anchors: set[str]) -> set[str]:
    """Final dotted component, e.g. mlsysim.physics.memory.calc_x -> calc_x."""
    return {a.rsplit(".", 1)[-1] for a in anchors}


def check() -> list[str]:
    sys.path.insert(0, str(DOCS.parent))
    anchors = documented_anchors()
    names = short(anchors)
    missing: list[str] = []

    physics = importlib.import_module("mlsysim.physics")
    for name in physics.__all__:
        obj = getattr(physics, name)
        # Aliases such as calc_training_time = dTime are documented under the
        # name of the function object they point to.
        if inspect.isfunction(obj) and obj.__name__ not in names:
            missing.append(f"mlsysim.physics.{name}  (defined in {obj.__module__})")

    solvers = importlib.import_module("mlsysim.solvers")
    for name in solvers.__all__:
        if name.startswith("Base"):
            continue
        if name not in names:
            missing.append(f"mlsysim.solvers.{name}")

    for pkg in REGISTRY_PACKAGES:
        mod = importlib.import_module(f"mlsysim.{pkg}")
        for name in getattr(mod, "__all__", []):
            if f"{pkg}.{name}" in SKIP:
                continue
            if name not in names:
                missing.append(f"mlsysim.{pkg}.{name}")
    return missing


def stamp_version() -> str:
    """Write config/sidebar-brand.html with the installed package version."""
    sys.path.insert(0, str(DOCS.parent))
    version = importlib.import_module("mlsysim").__version__
    template = (DOCS / "config" / "sidebar-brand.html.in").read_text(encoding="utf-8")
    (DOCS / "config" / "sidebar-brand.html").write_text(
        template.replace("__MLSYSIM_VERSION__", version), encoding="utf-8"
    )
    return version


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--check", action="store_true", help="check coverage only")
    args = parser.parse_args()

    # The sidebar include must exist for every render, including single pages.
    stamp_version()

    # As a pre-render hook, only rebuild the API on full-project renders so
    # that single-page previews stay fast.
    in_quarto = "QUARTO_PROJECT_RENDER_ALL" in os.environ
    if in_quarto and os.environ.get("QUARTO_PROJECT_RENDER_ALL") != "1":
        return 0

    if not args.check:
        build()

    missing = check()
    if missing:
        print("API coverage check failed. These public symbols have no API page:", file=sys.stderr)
        for item in missing:
            print(f"  - {item}", file=sys.stderr)
        print("Add them to the quartodoc block in mlsysim/docs/_quarto.yml.", file=sys.stderr)
        return 1
    print(f"API coverage OK: {len(documented_anchors())} documented objects.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

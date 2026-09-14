#!/usr/bin/env python
"""
Module Integration Tests for TinyTorch
======================================

Guards the boundary between the 20 source modules and the package they build.

This file previously tested a package layout that no longer exists
(`tinytorch.core.autograd.Variable`, `tinytorch.utils`, `tinytorch.profiler`).
Every one of its five tests failed, and every one reported success anyway,
because failure was signalled with `return False` -- which pytest discards.
Rewritten to check things that are true of the current package and that fail
loudly when they stop being true.

What it guards:
  1. Every module's export target imports, in module order.
  2. Every import shown in a module's "Where This Code Lives" block resolves.
     (This is the check that caught OlympicEvent silently not being exported.)
  3. Progressive disclosure: no module imports from a later-numbered module.
  4. Data flows across the module boundary: tensor -> layer -> loss -> backward.
"""

import re
import importlib
import pathlib

import numpy as np
import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"


def _module_files():
    """(number, name, path) for each of the 20 source modules, in order."""
    out = []
    for d in sorted(SRC.glob("[0-9][0-9]_*")):
        py = next(d.glob("[0-9]*.py"), None)
        if py is not None:
            out.append((int(d.name[:2]), d.name, py))
    return out


def _export_targets():
    """Map 'tinytorch.core.tensor' -> (1, '01_tensor') from each default_exp."""
    targets = {}
    for num, name, py in _module_files():
        m = re.search(r"^#\|\s*default_exp\s+([\w.]+)", py.read_text(), re.M)
        if m:
            targets[f"tinytorch.{m.group(1)}"] = (num, name)
    return targets


MODULES = _module_files()
TARGETS = _export_targets()


def test_all_modules_present():
    """All 20 modules exist and each declares an export target."""
    assert len(MODULES) == 20, f"Expected 20 source modules, found {len(MODULES)}"
    assert len(TARGETS) == 20, (
        f"Expected 20 export targets, found {len(TARGETS)}. A module is missing "
        f"its '#| default_exp' directive."
    )
    numbers = [n for n, _, _ in MODULES]
    assert numbers == list(range(1, 21)), f"Module numbering has a gap: {numbers}"


@pytest.mark.parametrize("target", sorted(TARGETS), ids=lambda t: t.split(".")[-1])
def test_export_target_imports(target):
    """Each module's built package imports cleanly."""
    importlib.import_module(target)


def test_documented_imports_resolve():
    """
    Every symbol a module tells students to import must actually be importable.

    A module's "Where This Code Lives" block shows a `from tinytorch... import`
    line. If a name there is missing from the built package, a student
    following the instructions gets an ImportError.
    """
    broken = []
    for num, name, py in MODULES:
        for line in py.read_text().splitlines():
            m = re.match(r"\s*from (tinytorch\.[\w.]+) import (.+?)\s*(?:#.*)?$", line)
            if not m:
                continue
            mod_path, names = m.group(1), m.group(2)
            if "(" in names or names.strip() == "*":
                continue
            try:
                mod = importlib.import_module(mod_path)
            except Exception as exc:  # noqa: BLE001 - reported, not swallowed
                broken.append(f"{name}: cannot import {mod_path} ({exc})")
                continue
            for sym in (s.strip() for s in names.split(",")):
                if sym and not hasattr(mod, sym):
                    broken.append(f"{name}: {mod_path} has no '{sym}'")

    assert not broken, "Documented imports that do not resolve:\n  " + "\n  ".join(broken)


def test_progressive_disclosure():
    """
    No module may import from a later-numbered module.

    Students work through 01 to 20 in order. A backwards dependency means a
    module needs something the reader has not built yet.
    """
    violations = []
    for num, name, py in MODULES:
        for i, line in enumerate(py.read_text().splitlines(), 1):
            m = re.match(r"\s*from (tinytorch\.[\w.]+) import ", line)
            if not m:
                continue
            target = m.group(1)
            if target in TARGETS and TARGETS[target][0] > num:
                violations.append(
                    f"{name}:{i} imports {target} (built in module "
                    f"{TARGETS[target][1]})"
                )
    assert not violations, "Backwards dependencies:\n  " + "\n  ".join(violations)


def test_cross_module_data_flow():
    """A tensor built in module 01 survives a round trip through 03, 04 and 06."""
    from tinytorch.core.tensor import Tensor
    from tinytorch.core.layers import Linear
    from tinytorch.core.losses import MSELoss
    from tinytorch.core.optimizers import SGD

    rng = np.random.default_rng(7)
    x = Tensor(rng.standard_normal((4, 3)).astype(np.float32))
    target = Tensor(rng.standard_normal((4, 2)).astype(np.float32))

    layer = Linear(3, 2)
    # Constructing the optimizer is what enables requires_grad on parameters.
    optimizer = SGD(layer.parameters(), lr=0.01)

    prediction = layer(x)
    assert prediction.shape == (4, 2), f"Linear reshaped wrongly: {prediction.shape}"

    loss = MSELoss()(prediction, target)
    assert loss._grad_fn is not None, (
        "Loss carries no _grad_fn, so backward() would silently do nothing"
    )

    loss.backward()
    assert layer.weight.grad is not None, "No gradient reached the layer weight"
    assert layer.bias.grad is not None, "No gradient reached the layer bias"

    before = np.array(layer.weight.data, copy=True)
    optimizer.step()
    assert not np.allclose(before, np.asarray(layer.weight.data)), (
        "optimizer.step() did not change the weight"
    )

"""Shared helper: load an exercise's ``starter`` or ``solution`` module.

Each test imports the implementation named by the ``STAFFML_IMPL`` environment
variable (``solution`` by default), so the same test file grades both:

    pytest staffml/exercises/kv-cache-decode                      # reference solution
    STAFFML_IMPL=starter pytest staffml/exercises/kv-cache-decode # your code
"""
import importlib.util
import os
from pathlib import Path


def load(exercise_dir: str):
    name = os.environ.get("STAFFML_IMPL", "solution")
    if name not in ("starter", "solution"):
        raise ValueError(f"STAFFML_IMPL must be 'starter' or 'solution', got {name!r}")
    path = Path(exercise_dir) / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{Path(exercise_dir).name}_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

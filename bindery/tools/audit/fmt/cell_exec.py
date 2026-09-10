"""Headless execution helpers for audit scripts that eval Quarto python cells."""
from __future__ import annotations
import os
from typing import Any

def setup_headless_matplotlib() -> None:
    os.environ["MPLBACKEND"] = "Agg"
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        plt.ioff()
        plt.show = lambda *_a, **_k: None  # type: ignore[method-assign]
    except ImportError:
        pass

def patch_show_in_namespace(ns: dict) -> None:
    plt = ns.get("plt")
    if plt is not None and hasattr(plt, "show"):
        plt.show = lambda *_a, **_k: None  # type: ignore[method-assign]
    try:
        import matplotlib.pyplot as _plt
        _plt.show = lambda *_a, **_k: None  # type: ignore[method-assign]
    except ImportError:
        pass

def make_exec_namespace() -> dict[str, Any]:
    setup_headless_matplotlib()
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[4]
    repo_str = str(repo)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    mlsysim = repo / "mlsysim"
    if mlsysim.is_dir() and str(mlsysim) not in sys.path:
        sys.path.insert(0, str(mlsysim))
    return {"__builtins__": __builtins__}

def exec_cell_code(code: str, ns: dict) -> None:
    setup_headless_matplotlib()
    exec(compile(code, "<audit-cell>", "exec"), ns)  # noqa: S102
    patch_show_in_namespace(ns)

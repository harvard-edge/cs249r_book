"""Headless execution helpers for audit scripts that eval Quarto python cells."""
from __future__ import annotations
import os
from typing import Any

def setup_headless_matplotlib() -> None:
    """Force the non-interactive Agg backend and turn ``plt.show`` into a no-op.

    Sets ``MPLBACKEND=Agg`` in the process environment, switches matplotlib to
    Agg, disables interactive mode, and replaces ``pyplot.show`` so cells that
    call it neither open windows nor block. Does nothing further when
    matplotlib is not installed.
    """
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
    """Re-stub ``show`` after a cell runs, since the cell may have rebound it.

    Replaces ``show`` with a no-op on the ``plt`` object in ``ns`` (if present)
    and on ``matplotlib.pyplot`` itself, skipping the latter when matplotlib is
    not installed.
    """
    plt = ns.get("plt")
    if plt is not None and hasattr(plt, "show"):
        plt.show = lambda *_a, **_k: None  # type: ignore[method-assign]
    try:
        import matplotlib.pyplot as _plt
        _plt.show = lambda *_a, **_k: None  # type: ignore[method-assign]
    except ImportError:
        pass

def make_exec_namespace() -> dict[str, Any]:
    """Return a fresh exec namespace containing only ``__builtins__``.

    Also configures headless matplotlib and prepends the repository root and
    its ``mlsysim/`` directory to ``sys.path`` (when not already present) so
    cells can import ``mlsysim``. The ``sys.path`` change persists for the
    process.
    """
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
    """Execute one cell's source in ``ns`` with headless matplotlib.

    The code is compiled under the filename ``<audit-cell>``; any exception the
    cell raises propagates to the caller. After a successful run, ``show`` is
    re-stubbed in case the cell rebound it.
    """
    setup_headless_matplotlib()
    exec(compile(code, "<audit-cell>", "exec"), ns)  # noqa: S102
    patch_show_in_namespace(ns)
